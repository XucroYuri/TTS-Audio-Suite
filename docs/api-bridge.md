# API Bridge external-runner interruption

The API Bridge observes a targeted ComfyUI interruption within 250 ms while an
external TTS runner is active. IndexTTS, GPT-SoVITS, and CosyVoice share the
same bounded wait and process-tree cleanup path. If cleanup cannot verify that
the runner tree exited, the Bridge reports cleanup failure instead of reporting
a false interruption success.
