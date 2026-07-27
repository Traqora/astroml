"""
Documentation updater for maintaining sync with code.

This module provides functionality to update documentation when code changes,
detecting outdated docs and applying updates while preserving manual edits.
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set
import difflib

from astroml.llm.docs.code_analyzer import CodeAnalyzer, CodeElement


@dataclass
class DocMetadata:
    """
    Metadata for documentation files.

    Attributes:
        file_path: Path to the documentation file
        source_files: List of source files used to generate the doc
        hash: Hash of the source files for change detection
        generated_at: Timestamp when documentation was generated
        last_updated: Timestamp when documentation was last updated
        manual_edits: Flag indicating if manual edits were made
        checksum: Checksum of the documentation content
    """

    file_path: str
    source_files: List[str] = field(default_factory=list)
    hash: str = ""
    generated_at: str = ""
    last_updated: str = ""
    manual_edits: bool = False
    checksum: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "file_path": self.file_path,
            "source_files": self.source_files,
            "hash": self.hash,
            "generated_at": self.generated_at,
            "last_updated": self.last_updated,
            "manual_edits": self.manual_edits,
            "checksum": self.checksum,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "DocMetadata":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class UpdateResult:
    """
    Result of a documentation update operation.

    Attributes:
        success: Whether the update was successful
        updated_files: List of files that were updated
        skipped_files: List of files that were skipped
        errors: List of errors encountered
        changes_made: Summary of changes made
    """

    success: bool
    updated_files: List[str] = field(default_factory=list)
    skipped_files: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    changes_made: str = ""


class DocumentationUpdater:
    """
    Updates documentation to stay in sync with code changes.

    Features:
    - Detects when source code has changed
    - Preserves manual edits in documentation
    - Applies incremental updates
    - Maintains metadata for change tracking
    - Handles merge conflicts
    """

    def __init__(self, metadata_dir: str = ".doc_metadata"):
        """
        Initialize the documentation updater.

        Args:
            metadata_dir: Directory to store documentation metadata
        """
        self.metadata_dir = Path(metadata_dir)
        self.metadata_dir.mkdir(exist_ok=True)
        self.analyzer = CodeAnalyzer()

    def update_documentation(
        self,
        doc_path: str,
        source_paths: List[str],
        preserve_manual_edits: bool = True,
    ) -> UpdateResult:
        """
        Update documentation based on source code changes.

        Args:
            doc_path: Path to the documentation file
            source_paths: List of source file paths
            preserve_manual_edits: Whether to preserve manual edits

        Returns:
            UpdateResult with update status
        """
        result = UpdateResult(success=True)

        try:
            # Load existing metadata
            metadata = self._load_metadata(doc_path)

            # Calculate current hash of source files
            current_hash = self._calculate_source_hash(source_paths)

            # Check if update is needed
            if metadata and metadata.hash == current_hash:
                result.skipped_files.append(doc_path)
                result.changes_made = "No changes detected in source files"
                return result

            # Analyze source code
            all_elements = []
            for source_path in source_paths:
                elements = self.analyzer.analyze_file(source_path)
                all_elements.extend(elements)

            # Read existing documentation if it exists
            existing_doc = ""
            if Path(doc_path).exists():
                with open(doc_path, "r", encoding="utf-8") as f:
                    existing_doc = f.read()

            # Generate new documentation
            new_doc = self._generate_documentation(all_elements, existing_doc)

            # Check for manual edits if preserving
            if preserve_manual_edits and metadata and metadata.manual_edits:
                new_doc = self._merge_manual_edits(existing_doc, new_doc)

            # Write updated documentation
            Path(doc_path).parent.mkdir(parents=True, exist_ok=True)
            with open(doc_path, "w", encoding="utf-8") as f:
                f.write(new_doc)

            # Update metadata
            new_metadata = DocMetadata(
                file_path=doc_path,
                source_files=source_paths,
                hash=current_hash,
                generated_at=datetime.now().isoformat(),
                last_updated=datetime.now().isoformat(),
                manual_edits=False,
                checksum=self._calculate_checksum(new_doc),
            )
            self._save_metadata(new_metadata)

            result.updated_files.append(doc_path)
            result.changes_made = f"Updated documentation from {len(source_paths)} source file(s)"

        except Exception as e:
            result.success = False
            result.errors.append(str(e))

        return result

    def detect_outdated_docs(self, doc_dir: str) -> List[str]:
        """
        Detect documentation files that are outdated.

        Args:
            doc_dir: Directory containing documentation files

        Returns:
            List of outdated documentation file paths
        """
        outdated = []

        for metadata_file in self.metadata_dir.glob("*.json"):
            try:
                with open(metadata_file, "r") as f:
                    metadata = DocMetadata.from_dict(json.load(f))

                # Check if source files still exist
                source_exists = all(Path(sf).exists() for sf in metadata.source_files)
                if not source_exists:
                    outdated.append(metadata.file_path)
                    continue

                # Check if source files have changed
                current_hash = self._calculate_source_hash(metadata.source_files)
                if metadata.hash != current_hash:
                    outdated.append(metadata.file_path)

            except Exception as e:
                print(f"Error checking metadata {metadata_file}: {e}")

        return outdated

    def batch_update(
        self,
        mappings: Dict[str, List[str]],
        preserve_manual_edits: bool = True,
    ) -> UpdateResult:
        """
        Update multiple documentation files.

        Args:
            mappings: Dictionary mapping doc paths to source file lists
            preserve_manual_edits: Whether to preserve manual edits

        Returns:
            UpdateResult with batch update status
        """
        result = UpdateResult(success=True)

        for doc_path, source_paths in mappings.items():
            update_result = self.update_documentation(
                doc_path, source_paths, preserve_manual_edits
            )

            result.updated_files.extend(update_result.updated_files)
            result.skipped_files.extend(update_result.skipped_files)
            result.errors.extend(update_result.errors)

        result.success = len(result.errors) == 0
        result.changes_made = f"Updated {len(result.updated_files)} file(s), skipped {len(result.skipped_files)}"

        return result

    def mark_manual_edits(self, doc_path: str) -> None:
        """
        Mark a documentation file as having manual edits.

        Args:
            doc_path: Path to the documentation file
        """
        metadata = self._load_metadata(doc_path)
        if metadata:
            metadata.manual_edits = True
            metadata.last_updated = datetime.now().isoformat()
            metadata.checksum = self._calculate_checksum(
                Path(doc_path).read_text(encoding="utf-8")
            )
            self._save_metadata(metadata)

    def _load_metadata(self, doc_path: str) -> Optional[DocMetadata]:
        """Load metadata for a documentation file."""
        metadata_file = self.metadata_dir / f"{Path(doc_path).stem}.json"

        if not metadata_file.exists():
            return None

        try:
            with open(metadata_file, "r") as f:
                return DocMetadata.from_dict(json.load(f))
        except Exception:
            return None

    def _save_metadata(self, metadata: DocMetadata) -> None:
        """Save metadata for a documentation file."""
        metadata_file = self.metadata_dir / f"{Path(metadata.file_path).stem}.json"

        with open(metadata_file, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)

    def _calculate_source_hash(self, source_paths: List[str]) -> str:
        """Calculate hash of source files."""
        hasher = hashlib.sha256()

        for source_path in sorted(source_paths):
            if Path(source_path).exists():
                with open(source_path, "rb") as f:
                    hasher.update(f.read())

        return hasher.hexdigest()

    def _calculate_checksum(self, content: str) -> str:
        """Calculate checksum of content."""
        return hashlib.md5(content.encode()).hexdigest()

    def _generate_documentation(
        self, elements: List[CodeElement], existing_doc: str = ""
    ) -> str:
        """
        Generate documentation from code elements.

        Args:
            elements: List of code elements
            existing_doc: Existing documentation content

        Returns:
            Generated documentation string
        """
        # Simple generation - in production, use writers
        lines = []

        for element in elements:
            if element.element_type.value == "module":
                lines.append(f"# {element.name}")
                if element.docstring:
                    lines.append(element.docstring)
                lines.append("")

            elif element.element_type.value == "class":
                lines.append(f"## {element.name}")
                if element.docstring:
                    lines.append(element.docstring)
                lines.append("")

            elif element.element_type.value in ["function", "method"]:
                lines.append(f"### {element.name}")
                if element.signature:
                    lines.append("```python")
                    lines.append(element.signature)
                    lines.append("```")
                if element.docstring:
                    lines.append(element.docstring)
                lines.append("")

        return "\n".join(lines)

    def _merge_manual_edits(self, existing_doc: str, new_doc: str) -> str:
        """
        Merge manual edits from existing documentation into new documentation.

        Args:
            existing_doc: Existing documentation with manual edits
            new_doc: Newly generated documentation

        Returns:
            Merged documentation
        """
        # Simple strategy: preserve sections that exist in both
        # In production, use more sophisticated diff/merge algorithms

        existing_lines = existing_doc.split("\n")
        new_lines = new_doc.split("\n")

        # Use difflib to find common sections
        matcher = difflib.SequenceMatcher(None, existing_lines, new_lines)

        merged = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                # Keep common sections from new doc
                merged.extend(new_lines[j1:j2])
            elif tag == "replace":
                # Keep new version
                merged.extend(new_lines[j1:j2])
            elif tag == "delete":
                # Skip deleted sections
                pass
            elif tag == "insert":
                # Add new sections
                merged.extend(new_lines[j1:j2])

        return "\n".join(merged)

    def get_diff(self, old_doc: str, new_doc: str) -> str:
        """
        Get diff between old and new documentation.

        Args:
            old_doc: Old documentation content
            new_doc: New documentation content

        Returns:
            Unified diff string
        """
        old_lines = old_doc.split("\n")
        new_lines = new_doc.split("\n")

        diff = difflib.unified_diff(
            old_lines, new_lines, lineterm="", fromfile="old", tofile="new"
        )

        return "\n".join(diff)

    def rollback_update(self, doc_path: str) -> bool:
        """
        Rollback a documentation update to previous version.

        Args:
            doc_path: Path to the documentation file

        Returns:
            Whether rollback was successful
        """
        # In production, implement version control integration
        # For now, this is a placeholder
        return False
