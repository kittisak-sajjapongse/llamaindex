from typing import Any, Dict, List, Optional
from llama_index.core.callbacks import CBEventType
from llama_index.core.callbacks.base_handler import BaseCallbackHandler 


class LLMTemplateLogger(BaseCallbackHandler):
    def __init__(self):
        super().__init__([], [])
        self.messages = None

    def on_event_start(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> str:
        if event_type == CBEventType.LLM and payload is not None:
            messages = payload.get("messages")
            if messages:
                self.messages = messages
        return event_id

    def get_messages(self):
        return self.messages

    def on_event_end(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]] = None,
        event_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        pass

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        pass

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        pass