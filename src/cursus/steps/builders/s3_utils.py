from pathlib import PurePosixPath
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class S3PathHandler:
    """
    Utility class for handling S3 URIs and paths.

    Provides methods to properly manipulate S3 paths while avoiding common issues
    like double slashes or incorrect path handling that can occur when using
    standard os.path functions with S3 URIs.
    """

    @staticmethod
    def parse_uri(uri: str) -> Tuple[str, str]:
        """
        Parse an S3 URI into components.

        Args:
            uri: S3 URI to parse

        Returns:
            tuple: (bucket, key)

        Raises:
            ValueError: If not a valid S3 URI
        """
        if not uri or not isinstance(uri, str) or not uri.startswith("s3://"):
            raise ValueError(f"Not a valid S3 URI: {uri}")

        parts = uri[5:].split("/", 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""

        return bucket, key

    @classmethod
    def normalize(cls, uri: str, description: str = "S3 URI") -> str:
        """
        Normalize an S3 URI by ensuring no trailing slashes or double slashes.

        Args:
            uri: S3 URI to normalize
            description: Description for logging

        Returns:
            Normalized S3 URI
        """
        if not uri or not isinstance(uri, str) or not uri.startswith("s3://"):
            return uri

        try:
            bucket, key = cls.parse_uri(uri)
        except ValueError:
            return uri

        # Clean up path segments
        if key:
            segments = [seg for seg in key.split("/") if seg]
            normalized_key = str(PurePosixPath(*segments)) if segments else ""
            normalized = f"s3://{bucket}/{normalized_key}"
        else:
            normalized = f"s3://{bucket}"

        # Log if changed
        if normalized != uri:
            logger.info(f"Normalized {description}: '{uri}' -> '{normalized}'")

        return normalized

    @classmethod
    def join(cls, base_uri: str, *paths: str) -> str:
        """
        Join path components with an S3 URI base.

        Args:
            base_uri: Base S3 URI
            *paths: Path components to join

        Returns:
            Joined S3 URI
        """
        if (
            not base_uri
            or not isinstance(base_uri, str)
            or not base_uri.startswith("s3://")
        ):
            return base_uri

        try:
            bucket, key = cls.parse_uri(base_uri)
        except ValueError:
            return base_uri

        # Collect path parts
        parts = []
        if key:
            parts.append(key)

        # Add additional parts
        parts.extend(p.lstrip("/") for p in paths if p)

        # Generate path using PurePosixPath
        if parts:
            path = str(PurePosixPath(*parts))
            return f"s3://{bucket}/{path}"
        else:
            return f"s3://{bucket}"

    @classmethod
    def get_parent(cls, uri: str) -> str:
        """
        Get the parent directory of an S3 URI.

        Args:
            uri: S3 URI

        Returns:
            Parent S3 URI
        """
        if not uri or not isinstance(uri, str) or not uri.startswith("s3://"):
            return uri

        try:
            bucket, key = cls.parse_uri(uri)
        except ValueError:
            return uri

        if not key:
            # At bucket root already
            return f"s3://{bucket}"

        # Use PurePosixPath for parent
        parent_key = str(PurePosixPath(key).parent)
        if parent_key == ".":
            return f"s3://{bucket}"

        return f"s3://{bucket}/{parent_key}"

    @classmethod
    def get_name(cls, uri: str) -> str:
        """
        Get the basename (filename) of an S3 URI.

        Args:
            uri: S3 URI

        Returns:
            Basename of the URI
        """
        if not uri or not isinstance(uri, str) or not uri.startswith("s3://"):
            return uri.split("/")[-1] if "/" in uri else uri

        try:
            bucket, key = cls.parse_uri(uri)
        except ValueError:
            return uri.split("/")[-1] if "/" in uri else uri

        if not key:
            return bucket

        return PurePosixPath(key).name

    @classmethod
    def ensure_directory(cls, uri: str, filename: str = None) -> str:
        """
        Ensure URI represents a directory path (without filename).

        Args:
            uri: S3 URI
            filename: Optional filename to check for at end

        Returns:
            Directory path without trailing filename
        """
        # First normalize
        uri = cls.normalize(uri)

        if not filename:
            return uri

        # Check if path ends with filename
        basename = cls.get_name(uri)
        if basename == filename:
            return cls.get_parent(uri)

        return uri

    @classmethod
    def is_valid(cls, uri: str) -> bool:
        """
        Check if a string is a valid S3 URI.

        Args:
            uri: URI to validate

        Returns:
            True if valid S3 URI
        """
        if not uri or not isinstance(uri, str) or not uri.startswith("s3://"):
            return False

        try:
            cls.parse_uri(uri)
            return True
        except ValueError:
            return False
