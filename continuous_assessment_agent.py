
import random
import time
from datetime import datetime
from dataclasses import dataclass, field


@dataclass
class Question:
    question_id  : str
    concept_id   : str
    difficulty   : str           
    text         : str
    options      : list[str]
    correct_index: int            
    explanation  : str = ""


@dataclass
class Response:
    question_id    : str
    concept_id     : str
    chosen_index   : int
    is_correct     : bool
    time_taken_sec : float
    timestamp      : str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class AssessmentResult:
    learner_id     : str
    session_id     : str
    concept_id     : str
    difficulty     : str
    total_questions: int
    correct        : int
    score_pct      : float
    responses      : list[Response]
    misconceptions : list[str]
    timestamp      : str = field(default_factory=lambda: datetime.now().isoformat())

    def to_lma_payload(self) -> dict:
        """Format result for the Learner Modeling Agent."""
        return {
            "learner_id": self.learner_id,
            "responses": [
                {
                    "concept_id": r.concept_id,
                    "correct"   : r.is_correct,
                    "time_sec"  : r.time_taken_sec,
                }
                for r in self.responses
            ],
        }


QUESTION_BANK: dict[str, list[Question]] = {
    "py_vars": [
        Question(
            "v_e1", "py_vars", "easy",
            "Which keyword declares a variable in Python?",
            ["var", "let", "No keyword needed", "dim"],
            2,
            "Python is dynamically typed — no keyword required.",
        ),
        Question(
            "v_m1", "py_vars", "medium",
            "What is the output of: x = 5; x = x + 2.0; print(type(x))?",
            ["<class 'int'>", "<class 'float'>", "<class 'str'>", "TypeError"],
            1,
            "Adding an int and a float promotes the result to float.",
        ),
        Question(
            "v_h1", "py_vars", "hard",
            "Which of these creates an immutable sequence in Python?",
            ["list", "dict", "tuple", "set"],
            2,
            "Tuples are immutable; lists, dicts, and sets are mutable.",
        ),
    ],
    "py_loops": [
        Question(
            "l_e1", "py_loops", "easy",
            "Which keyword exits a loop immediately?",
            ["exit", "stop", "break", "end"],
            2,
            "'break' terminates the nearest enclosing loop.",
        ),
        Question(
            "l_m1", "py_loops", "medium",
            "How many times does this print?\n  for i in range(2, 10, 3): print(i)",
            ["2", "3", "4", "8"],
            1,
            "range(2,10,3) → 2, 5, 8 → three values.",
        ),
        Question(
            "l_h1", "py_loops", "hard",
            "What does 'continue' do inside a loop?",
            [
                "Exits the loop",
                "Skips rest of current iteration",
                "Restarts the loop from beginning",
                "Pauses execution",
            ],
            1,
            "'continue' skips the remaining body of the current iteration.",
        ),
    ],
    "py_funcs": [
        Question(
            "f_e1", "py_funcs", "easy",
            "Which keyword defines a function in Python?",
            ["func", "define", "def", "function"],
            2,
            "Python uses 'def' to declare a function.",
        ),
        Question(
            "f_m1", "py_funcs", "medium",
            "What is returned when a function has no return statement?",
            ["0", "False", "None", "Error"],
            2,
            "Python implicitly returns None when there's no return.",
        ),
        Question(
            "f_h1", "py_funcs", "hard",
            "What is a closure in Python?",
            [
                "A class with private methods",
                "A function that remembers the enclosing scope",
                "A locked dictionary",
                "A decorator pattern",
            ],
            1,
            "A closure is a nested function that captures variables "
            "from its enclosing scope.",
        ),
    ],
}

MISCONCEPTION_MAP: dict[tuple[str, int], str] = {
    ("v_e1", 0): "Confused Python with JavaScript (var/let)",
    ("v_e1", 1): "Confused Python with JavaScript (var/let)",
    ("v_m1", 0): "Missed implicit type promotion (int + float = float)",
    ("l_m1", 0): "Miscounted range steps",
    ("l_m1", 3): "Used last value of range instead of count",
    ("f_m1", 0): "Assumed default return value is 0",
    ("f_m1", 1): "Assumed default return value is False",
}

class ContinuousAssessmentAgent:
    """
    AGENT 2 — Continuous Assessment Agent (CAA)

    Responsibilities:
      • Generate adaptive quizzes per concept and difficulty
      • Evaluate learner responses in real time
      • Detect misconceptions from wrong-answer patterns
      • Forward results to Learner Modeling Agent
    """

    DIFFICULTY_ORDER = ["easy", "medium", "hard"]

    def __init__(self):
        self.session_counter = 0
        self.outbox: list[dict] = []    # forwarded to LMA

    def generate_quiz(self, concept_id: str,
                      difficulty: str = "medium",
                      n_questions: int = 3) -> list[Question]:
        """
        Select questions from the bank for the given concept + difficulty.
        Falls back to all difficulties if not enough questions exist.
        """
        pool = QUESTION_BANK.get(concept_id, [])
        filtered = [q for q in pool if q.difficulty == difficulty]
        if len(filtered) < n_questions:
            filtered = pool          # fall back to mixed difficulty

        selected = random.sample(filtered, min(n_questions, len(filtered)))
        print(f"\n[CAA] 📝 Quiz generated | Concept: '{concept_id}' | "
              f"Difficulty: {difficulty} | Questions: {len(selected)}")
        return selected

    def evaluate_responses(
        self,
        learner_id : str,
        concept_id : str,
        difficulty : str,
        questions  : list[Question],
        answers    : list[int],          # learner's chosen option indices
        times      : list[float],        # time per question in seconds
    ) -> AssessmentResult:
        """
        Score the quiz and detect misconceptions.

        answers and times must align positionally with questions.
        """
        self.session_counter += 1
        session_id = f"SES{self.session_counter:04d}"

        responses      : list[Response] = []
        misconceptions : list[str]      = []
        correct_count                   = 0

        for q, chosen, t in zip(questions, answers, times):
            is_correct = (chosen == q.correct_index)
            if is_correct:
                correct_count += 1
            else:
                # Check for known misconception
                key = (q.question_id, chosen)
                if key in MISCONCEPTION_MAP:
                    misconceptions.append(MISCONCEPTION_MAP[key])

            responses.append(Response(
                question_id    = q.question_id,
                concept_id     = q.concept_id,
                chosen_index   = chosen,
                is_correct     = is_correct,
                time_taken_sec = t,
            ))

        score_pct = round(correct_count / len(questions) * 100, 1)

        result = AssessmentResult(
            learner_id      = learner_id,
            session_id      = session_id,
            concept_id      = concept_id,
            difficulty      = difficulty,
            total_questions = len(questions),
            correct         = correct_count,
            score_pct       = score_pct,
            responses       = responses,
            misconceptions  = list(set(misconceptions)),
        )

        print(f"[CAA] 📊 Evaluated Session {session_id} | "
              f"Score: {correct_count}/{len(questions)} ({score_pct}%)")
        if misconceptions:
            print(f"[CAA] ⚠️  Misconceptions found: {list(set(misconceptions))}")

        return result

    def adapt_difficulty(self, current: str, score_pct: float) -> str:
        """Bump difficulty up/down based on score."""
        idx = self.DIFFICULTY_ORDER.index(current)
        if score_pct >= 80 and idx < 2:
            new = self.DIFFICULTY_ORDER[idx + 1]
            print(f"[CAA] ⬆  Score {score_pct}% — upgrading difficulty "
                  f"'{current}' → '{new}'")
            return new
        elif score_pct < 50 and idx > 0:
            new = self.DIFFICULTY_ORDER[idx - 1]
            print(f"[CAA] ⬇  Score {score_pct}% — downgrading difficulty "
                  f"'{current}' → '{new}'")
            return new
        return current
    def send_to_lma(self, result: AssessmentResult) -> dict:
        """Package and forward the assessment result to LMA."""
        message = {
            "source"     : "ContinuousAssessmentAgent",
            "destination": "LearnerModelingAgent",
            "timestamp"  : datetime.now().isoformat(),
            "payload"    : result.to_lma_payload(),
            "metadata"   : {
                "session_id"    : result.session_id,
                "score_pct"     : result.score_pct,
                "misconceptions": result.misconceptions,
            },
        }
        self.outbox.append(message)
        print(f"[CAA] 📤 Results forwarded to Learner Modeling Agent.")
        return message

    def run_assessment(
        self,
        learner_id  : str,
        concept_id  : str,
        difficulty  : str = "medium",
        n_questions : int = 3,
        answers     : list[int] | None = None,  
        times       : list[float] | None = None,
    ) -> AssessmentResult:
        """
        End-to-end: generate quiz → evaluate → adapt difficulty → emit.
        If answers is None, random answers are simulated.
        """
        questions = self.generate_quiz(concept_id, difficulty, n_questions)

        if answers is None:
           
            answers = [
                q.correct_index if random.random() < 0.65
                else random.choice([i for i in range(4) if i != q.correct_index])
                for q in questions
            ]
        if times is None:
            times = [round(random.uniform(15, 60), 1) for _ in questions]

    
        print("\n── Quiz ─────────────────────────────────────────")
        for i, (q, a) in enumerate(zip(questions, answers), 1):
            correct_mark = "✅" if a == q.correct_index else "❌"
            print(f"  Q{i}: {q.text}")
            print(f"       Learner chose: [{a}] {q.options[a]}  {correct_mark}")
            if a != q.correct_index:
                print(f"       Correct answer: [{q.correct_index}] "
                      f"{q.options[q.correct_index]}")
                print(f"       💡 {q.explanation}")
        print("─────────────────────────────────────────────────")

        result = self.evaluate_responses(
            learner_id, concept_id, difficulty, questions, answers, times
        )
     
        self.adapt_difficulty(difficulty, result.score_pct)
   
        self.send_to_lma(result)
        return result


def run_demo():
    print("\n" + "="*55)
    print("  AGENT 2: Continuous Assessment Agent — Demo")
    print("  Team INFINYX | DevHack 2026 | SKCT")
    print("="*55 + "\n")

    caa = ContinuousAssessmentAgent()
    learner_id = "L001"

  
    print("══ SESSION 1 — Variables | Easy ════════════════")
    r1 = caa.run_assessment(learner_id, "py_vars", difficulty="easy")

  
    print("\n══ SESSION 2 — Loops | Medium ══════════════════")
    r2 = caa.run_assessment(learner_id, "py_loops", difficulty="medium")

    print("\n══ SESSION 3 — Functions | Hard (struggling) ═══")
    r3 = caa.run_assessment(
        learner_id, "py_funcs", difficulty="hard",
        answers=[0, 0, 0],  
        times=[70, 80, 90],
    )

    print("\n" + "="*55)
    print("  ASSESSMENT SUMMARY")
    print("="*55)
    for i, r in enumerate([r1, r2, r3], 1):
        print(f"  Session {i} | {r.concept_id:<12} | "
              f"Score: {r.correct}/{r.total_questions} ({r.score_pct}%) | "
              f"Misconceptions: {len(r.misconceptions)}")
    print("="*55)
    print(f"\n  Outbox messages sent to LMA: {len(caa.outbox)}\n")


if __name__ == "__main__":
    run_demo()