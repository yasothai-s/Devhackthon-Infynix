
import json
import random
from datetime import datetime
from collections import defaultdict


class ConceptNode:
    """Represents a single concept in the learner's knowledge graph."""

    def __init__(self, concept_id: str, name: str):
        self.concept_id   = concept_id
        self.name         = name
        self.mastery      = 0.0          # 0.0 → 1.0
        self.attempts     = 0
        self.correct      = 0
        self.last_seen    = None
        self.prerequisites = []          # list of concept_ids

    def update(self, is_correct: bool):
        """Bayesian Knowledge Tracing — simplified update."""
        self.attempts  += 1
        self.last_seen  = datetime.now().isoformat()
        if is_correct:
            self.correct += 1
      
        raw_accuracy = self.correct / self.attempts
        self.mastery = round(min(raw_accuracy + 0.05, 1.0), 3)

    def is_mastered(self, threshold: float = 0.75) -> bool:
        return self.mastery >= threshold

    def to_dict(self) -> dict:
        return {
            "concept_id"   : self.concept_id,
            "name"         : self.name,
            "mastery"      : self.mastery,
            "attempts"     : self.attempts,
            "correct"      : self.correct,
            "last_seen"    : self.last_seen,
            "mastered"     : self.is_mastered(),
        }


class LearnerProfile:
    """Stores the complete dynamic profile of a single learner."""

    LEARNING_STYLES = ["visual", "auditory", "reading", "kinesthetic"]

    def __init__(self, learner_id: str, name: str):
        self.learner_id      = learner_id
        self.name            = name
        self.knowledge_graph : dict[str, ConceptNode] = {}
        self.learning_style  = random.choice(self.LEARNING_STYLES)
        self.avg_pace        = 1.0          
        self.session_times   : list[float] = []
        self.created_at      = datetime.now().isoformat()
        self.last_updated    = None

    def add_concept(self, concept_id: str, name: str,
                    prerequisites: list[str] = None):
        node = ConceptNode(concept_id, name)
        node.prerequisites = prerequisites or []
        self.knowledge_graph[concept_id] = node

    def record_response(self, concept_id: str, is_correct: bool,
                        time_taken_sec: float = 30.0):
        """Update the knowledge graph after one quiz response."""
        if concept_id not in self.knowledge_graph:
            raise ValueError(f"Unknown concept: {concept_id}")

        self.knowledge_graph[concept_id].update(is_correct)
        self.session_times.append(time_taken_sec)
      
        if len(self.session_times) >= 3:
            avg = sum(self.session_times[-5:]) / min(5, len(self.session_times))
            self.avg_pace = round(30.0 / avg, 2)   
        self.last_updated = datetime.now().isoformat()

    def detect_gaps(self) -> list[str]:
        """Return concept IDs where mastery is below threshold."""
        return [
            cid for cid, node in self.knowledge_graph.items()
            if not node.is_mastered()
        ]

    def strengths(self) -> list[str]:
        return [
            cid for cid, node in self.knowledge_graph.items()
            if node.is_mastered()
        ]

    def summary(self) -> dict:
        return {
            "learner_id"    : self.learner_id,
            "name"          : self.name,
            "learning_style": self.learning_style,
            "avg_pace"      : self.avg_pace,
            "strengths"     : self.strengths(),
            "gaps"          : self.detect_gaps(),
            "knowledge_graph": {
                cid: node.to_dict()
                for cid, node in self.knowledge_graph.items()
            },
            "last_updated"  : self.last_updated,
        }


class LearnerModelingAgent:
    """
    AGENT 1 — Learner Modeling Agent (LMA)

    Responsibilities:
      • Initialise and maintain learner profiles
      • Process assessment data from Continuous Assessment Agent
      • Detect conceptual gaps via knowledge graph
      • Emit updated profile to Adaptive Learning Path Agent
    """

    def __init__(self):
        self.profiles : dict[str, LearnerProfile] = {}
        self.inbox    : list[dict] = []   
        self.outbox   : list[dict] = []  

    def register_learner(self, learner_id: str, name: str,
                         curriculum: list[dict]):
        """
        Register a new learner and seed their knowledge graph.

        curriculum example:
          [{"id": "py_vars", "name": "Python Variables",
            "prerequisites": []}, ...]
        """
        profile = LearnerProfile(learner_id, name)
        for concept in curriculum:
            profile.add_concept(
                concept["id"],
                concept["name"],
                concept.get("prerequisites", []),
            )
        self.profiles[learner_id] = profile
        print(f"[LMA] ✅ Registered learner '{name}' (ID: {learner_id}) "
              f"with {len(curriculum)} concepts.")
        return profile

   
    def receive_assessment(self, assessment_result: dict):
        """
        Called by Continuous Assessment Agent (CAA) after each quiz.

        assessment_result schema:
          {
            "learner_id": str,
            "responses": [
               {"concept_id": str, "correct": bool, "time_sec": float},
               ...
            ]
          }
        """
        self.inbox.append(assessment_result)
        learner_id = assessment_result["learner_id"]

        if learner_id not in self.profiles:
            print(f"[LMA] ⚠️  Unknown learner '{learner_id}'. Skipping.")
            return

        profile = self.profiles[learner_id]
        for resp in assessment_result["responses"]:
            try:
                profile.record_response(
                    resp["concept_id"],
                    resp["correct"],
                    resp.get("time_sec", 30.0),
                )
            except ValueError as e:
                print(f"[LMA] ⚠️  {e}")

        gaps = profile.detect_gaps()
        print(f"[LMA] 📊 Updated profile for '{profile.name}'. "
              f"Gaps detected: {gaps or 'None'}")

     
        self._emit_profile(profile)

    def _emit_profile(self, profile: LearnerProfile):
        """Send updated learner profile to Adaptive Learning Path Agent."""
        message = {
            "source"     : "LearnerModelingAgent",
            "destination": "AdaptiveLearningPathAgent",
            "timestamp"  : datetime.now().isoformat(),
            "payload"    : profile.summary(),
        }
        self.outbox.append(message)
        print(f"[LMA] 📤 Profile emitted to Path Planning Agent.")
        return message

    def get_profile(self, learner_id: str) -> dict | None:
        if learner_id in self.profiles:
            return self.profiles[learner_id].summary()
        return None

    def print_profile(self, learner_id: str):
        profile = self.get_profile(learner_id)
        if profile:
            print("\n" + "="*55)
            print(f"  LEARNER PROFILE — {profile['name']}")
            print("="*55)
            print(f"  Learning Style : {profile['learning_style']}")
            print(f"  Learning Pace  : {profile['avg_pace']}x")
            print(f"  Strengths      : {profile['strengths'] or '—'}")
            print(f"  Gaps           : {profile['gaps'] or '—'}")
            print("\n  Knowledge Graph:")
            for cid, node in profile["knowledge_graph"].items():
                bar = "█" * int(node["mastery"] * 10)
                bar = bar.ljust(10, "░")
                status = "✅" if node["mastered"] else "❌"
                print(f"    {status} {node['name']:<25} "
                      f"[{bar}] {node['mastery']*100:.0f}%")
            print("="*55 + "\n")
        else:
            print(f"[LMA] No profile found for '{learner_id}'.")

def run_demo():
    print("\n" + "="*55)
    print("  AGENT 1: Learner Modeling Agent — Demo")
    print("  Team INFINYX | DevHack 2026 | SKCT")
    print("="*55 + "\n")

    lma = LearnerModelingAgent()

    curriculum = [
        {"id": "py_vars",   "name": "Variables & Data Types",  "prerequisites": []},
        {"id": "py_loops",  "name": "Loops (for / while)",     "prerequisites": ["py_vars"]},
        {"id": "py_funcs",  "name": "Functions & Scope",       "prerequisites": ["py_loops"]},
        {"id": "py_oop",    "name": "OOP & Classes",           "prerequisites": ["py_funcs"]},
        {"id": "py_files",  "name": "File I/O",                "prerequisites": ["py_funcs"]},
    ]

    lma.register_learner("L001", "Aisha Kumar", curriculum)

    round1 = {
        "learner_id": "L001",
        "responses": [
            {"concept_id": "py_vars",  "correct": True,  "time_sec": 18.0},
            {"concept_id": "py_loops", "correct": True,  "time_sec": 25.0},
            {"concept_id": "py_funcs", "correct": False, "time_sec": 55.0},
            {"concept_id": "py_oop",   "correct": False, "time_sec": 70.0},
            {"concept_id": "py_files", "correct": True,  "time_sec": 20.0},
        ]
    }

    round2 = {
        "learner_id": "L001",
        "responses": [
            {"concept_id": "py_funcs", "correct": True,  "time_sec": 40.0},
            {"concept_id": "py_oop",   "correct": False, "time_sec": 65.0},
        ]
    }

    print("── Round 1 Assessment ──────────────────────────")
    lma.receive_assessment(round1)

    print("\n── Round 2 Assessment ──────────────────────────")
    lma.receive_assessment(round2)

    lma.print_profile("L001")

    print("── Outbox (latest message to Path Agent) ───────")
    latest = lma.outbox[-1]
    gaps   = latest["payload"]["gaps"]
    print(f"  Gaps forwarded : {gaps}")
    print(f"  Timestamp      : {latest['timestamp']}")
    print("────────────────────────────────────────────────\n")


if __name__ == "__main__":
    run_demo()