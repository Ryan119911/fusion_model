from typing import Callable, Optional


SCRIPT_MODES = ("as_is", "traditional")


class CharacterScriptMapper:
    """Normalize annotation labels to the script used by trajectory identities."""

    def __init__(
        self,
        mode: str = "traditional",
        converter: Optional[Callable[[str], str]] = None,
    ):
        if mode not in SCRIPT_MODES:
            raise ValueError(f"Unsupported character script mode {mode!r}")
        self.mode = mode
        if converter is not None:
            self._converter = converter
        elif mode == "as_is":
            self._converter = lambda value: value
        else:
            try:
                from opencc import OpenCC
            except ImportError as error:
                raise RuntimeError(
                    "Traditional target matching requires OpenCC. Install it with: "
                    "python -m pip install opencc-python-reimplemented"
                ) from error
            opencc = OpenCC("s2t")
            self._converter = opencc.convert

    def convert(self, label: str) -> str:
        value = str(label)
        converted = str(self._converter(value))
        if len(value) == 1 and len(converted) != 1:
            raise ValueError(
                f"Single-character label {value!r} converted to {converted!r}; "
                "use --target_script as_is or provide an unambiguous annotation"
            )
        return converted
