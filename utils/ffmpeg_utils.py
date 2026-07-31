"""
FFmpeg Utility Module
Centralized ffmpeg availability detection and error handling
"""

import subprocess
import logging
import os
from typing import Optional, List, Dict, Any

logger = logging.getLogger("TTS.FFmpeg")

class FFmpegUtils:
    """Centralized ffmpeg utility with graceful error handling"""

    _ffmpeg_available = None
    _ffmpeg_path = None
    _ffprobe_available = None
    _ffprobe_path = None
    _check_performed = False
    _ffprobe_check_performed = False
    _toolchain_validated = False
    _toolchain_ok = False

    @classmethod
    def is_available(cls) -> bool:
        """Check if ffmpeg is available in system PATH"""
        if cls._check_performed:
            return cls._ffmpeg_available

        cls._check_performed = True

        try:
            result = subprocess.run(['ffmpeg', '-version'],
                                  capture_output=True,
                                  timeout=5)
            cls._ffmpeg_available = result.returncode == 0
            if cls._ffmpeg_available:
                cls._ffmpeg_path = 'ffmpeg'
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            cls._ffmpeg_available = False
            cls._ffmpeg_path = None

        return cls._ffmpeg_available

    @classmethod
    def get_path(cls) -> Optional[str]:
        """Get ffmpeg executable path if available"""
        cls.is_available()  # Ensure check is performed
        return cls._ffmpeg_path

    @classmethod
    def is_ffprobe_available(cls) -> bool:
        """Check if ffprobe is available in system PATH.

        Separate from is_available() because F5-TTS preprocessing
        (pydub.AudioSegment.from_file) requires both ffmpeg AND ffprobe.
        """
        if cls._ffprobe_check_performed:
            return cls._ffprobe_available

        cls._ffprobe_check_performed = True

        try:
            result = subprocess.run(['ffprobe', '-version'],
                                  capture_output=True,
                                  timeout=5)
            cls._ffprobe_available = result.returncode == 0
            if cls._ffprobe_available:
                cls._ffprobe_path = 'ffprobe'
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            cls._ffprobe_available = False
            cls._ffprobe_path = None

        return cls._ffprobe_available

    @classmethod
    def is_audio_toolchain_available(cls) -> bool:
        """Check if the complete audio toolchain (ffmpeg + ffprobe) is ready.

        F5-TTS reference-audio preprocessing uses pydub, which shells out to
        BOTH ffmpeg and ffprobe. Missing either will cause silent one-second
        zero-tensor outputs that look like successful synthesis.

        Returns True only when both binaries are reachable on PATH.
        """
        if cls._toolchain_validated:
            return cls._toolchain_ok

        cls._toolchain_validated = True
        ffmpeg_ok = cls.is_available()
        ffprobe_ok = cls.is_ffprobe_available()
        cls._toolchain_ok = ffmpeg_ok and ffprobe_ok
        return cls._toolchain_ok

    @classmethod
    def require_audio_toolchain(cls, feature: str = "audio processing") -> None:
        """Raise actionable error if ffmpeg or ffprobe is missing.

        Call this at the earliest point before any pydub/AudioSegment call,
        so users get a clear fix instruction instead of silent zero audio.

        Args:
            feature: Human-readable feature name for the error message.
        Raises:
            RuntimeError with platform-specific install instructions.
        """
        if cls.is_audio_toolchain_available():
            return

        cls.is_available()       # populate _ffmpeg_available
        cls.is_ffprobe_available()  # populate _ffprobe_available

        missing = []
        if not cls._ffmpeg_available:
            missing.append("ffmpeg")
        if not cls._ffprobe_available:
            missing.append("ffprobe")

        msg = (
            f"{feature} requires both ffmpeg and ffprobe on PATH. "
            f"Missing: {', '.join(missing)}.\n"
            f"Install with:\n"
            f"  Windows: winget install --id Gyan.FFmpeg --exact\n"
            f"  macOS:   brew install ffmpeg\n"
            f"  Linux:   sudo apt-get install ffmpeg  (or dnf install ffmpeg)\n"
            f"Then restart ComfyUI."
        )
        raise RuntimeError(msg)

    @classmethod
    def run_command(cls, cmd_args: List[str], **kwargs) -> subprocess.CompletedProcess:
        """
        Run ffmpeg command with error handling

        Args:
            cmd_args: Command arguments (should start with input/output files)
            **kwargs: Additional subprocess.run arguments

        Returns:
            CompletedProcess result

        Raises:
            RuntimeError: If ffmpeg is not available
        """
        if not cls.is_available():
            raise RuntimeError(
                "FFmpeg is not installed or not in system PATH. "
                "Please install ffmpeg to use this feature. "
                "See: https://ffmpeg.org/download.html"
            )

        full_cmd = ['ffmpeg'] + cmd_args

        try:
            return subprocess.run(full_cmd, **kwargs)
        except Exception as e:
            raise RuntimeError(f"FFmpeg command failed: {e}")

    @classmethod
    def convert_to_mp3(cls, input_path: str, output_path: str,
                      bitrate: str = '320k', overwrite: bool = True) -> bool:
        """
        Convert audio file to MP3 format

        Args:
            input_path: Input audio file path
            output_path: Output MP3 file path
            bitrate: Audio bitrate (default: 320k)
            overwrite: Whether to overwrite existing files

        Returns:
            True if successful, False if ffmpeg unavailable

        Raises:
            RuntimeError: If conversion fails (but ffmpeg is available)
        """
        if not cls.is_available():
            logger.warning("⚠️ FFmpeg not available - MP3 conversion skipped")
            return False

        args = ['-i', input_path, '-b:a', bitrate]
        if overwrite:
            args.append('-y')
        args.append(output_path)

        try:
            result = cls.run_command(args, capture_output=True)
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg conversion failed: {result.stderr.decode()}")
            return True
        except Exception as e:
            logger.error(f"❌ FFmpeg MP3 conversion failed: {e}")
            raise

    @classmethod
    def get_audio_info(cls, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Get audio file information using ffprobe

        Returns:
            Dict with audio info or None if ffmpeg unavailable
        """
        if not cls.is_available():
            return None

        try:
            # Use ffprobe (part of ffmpeg) to get audio info
            result = subprocess.run([
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_streams', file_path
            ], capture_output=True, text=True, timeout=10)

            if result.returncode == 0:
                import json
                data = json.loads(result.stdout)
                # Extract audio stream info
                for stream in data.get('streams', []):
                    if stream.get('codec_type') == 'audio':
                        return {
                            'duration': float(stream.get('duration', 0)),
                            'sample_rate': int(stream.get('sample_rate', 0)),
                            'channels': int(stream.get('channels', 0)),
                            'codec': stream.get('codec_name', 'unknown')
                        }
        except Exception as e:
            logger.debug(f"FFprobe failed: {e}")

        return None

    @classmethod
    def get_status_info(cls) -> Dict[str, Any]:
        """Get ffmpeg status for logging/debugging"""
        return {
            'available': cls.is_available(),
            'path': cls.get_path(),
            'checked': cls._check_performed
        }

# Global convenience functions
def is_ffmpeg_available() -> bool:
    """Check if ffmpeg is available"""
    return FFmpegUtils.is_available()

def convert_to_mp3_safe(input_path: str, output_path: str,
                       fallback_format: str = 'wav') -> tuple[str, bool]:
    """
    Safely convert to MP3 with fallback

    Returns:
        (actual_output_path, used_mp3)
    """
    if FFmpegUtils.is_available():
        try:
            FFmpegUtils.convert_to_mp3(input_path, output_path)
            return output_path, True
        except Exception as e:
            logger.warning(f"⚠️ MP3 conversion failed: {e}")

    # Fallback to original format or specified fallback
    if fallback_format == 'wav' and not output_path.endswith('.wav'):
        fallback_path = output_path.rsplit('.', 1)[0] + '.wav'
        logger.info(f"💡 Using WAV format instead of MP3: {fallback_path}")
        return fallback_path, False

    return input_path, False

def log_ffmpeg_status():
    """Log ffmpeg availability status"""
    status = FFmpegUtils.get_status_info()
    if status['available']:
        logger.info(f"✅ FFmpeg available at: {status['path']}")
    else:
        logger.warning("⚠️ FFmpeg not found - some features will use fallback formats")
        logger.info("💡 Install FFmpeg for MP3 support: https://ffmpeg.org/download.html")