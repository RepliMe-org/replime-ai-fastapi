from services import AIService


def get_ai_controller() -> "AIController":
    """Dependency factory for AIController."""
    return AIController()


class AIController:
    def __init__(self):
        self.service = AIService()

    def health(self):
        return self.service.health_check()
