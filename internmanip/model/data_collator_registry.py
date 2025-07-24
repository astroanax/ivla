

class DataCollatorRegistry:
    collaters = {}

    @classmethod
    def register(cls, type: str):
        """
        Register a collator class.
        """
        def decorator(model_class):
            cls.collaters[type] = model_class

        return decorator

    @classmethod
    def register_fn(cls, type: str, model_class):
        cls.collaters[type] = model_class


    @classmethod
    def get_collator(cls, type: str):
        if not type in cls.collaters:
            return None
        return cls.collaters[type]