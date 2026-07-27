"""
Code analyzer for documentation generation.

This module provides AST-based code analysis to extract structure,
docstrings, type hints, and other metadata for documentation generation.
"""

import ast
import inspect
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Dict, Optional, Any, Set
import importlib.util


class ElementType(Enum):
    """Types of code elements."""

    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    VARIABLE = "variable"
    CONSTANT = "constant"
    PROPERTY = "property"
    DECORATOR = "decorator"


@dataclass
class CodeElement:
    """
    Represents a code element extracted for documentation.

    Attributes:
        name: Name of the element
        element_type: Type of the element
        docstring: Docstring content
        file_path: Path to the file containing the element
        line_number: Line number where element is defined
        signature: Function/class signature
        type_hints: Type hints for parameters and return
        decorators: List of decorators
        parameters: Function parameters
        returns: Return type information
        raises: Exceptions raised
        examples: Code examples found in docstring
        parent: Parent element (if nested)
        children: Child elements
        metadata: Additional metadata
    """

    name: str
    element_type: ElementType
    docstring: Optional[str] = None
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    signature: Optional[str] = None
    type_hints: Dict[str, str] = field(default_factory=dict)
    decorators: List[str] = field(default_factory=list)
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    returns: Optional[str] = None
    raises: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    parent: Optional["CodeElement"] = None
    children: List["CodeElement"] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "element_type": self.element_type.value,
            "docstring": self.docstring,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "signature": self.signature,
            "type_hints": self.type_hints,
            "decorators": self.decorators,
            "parameters": self.parameters,
            "returns": self.returns,
            "raises": self.raises,
            "examples": self.examples,
            "metadata": self.metadata,
        }


class CodeAnalyzer:
    """
    Analyzes code structure and extracts documentation-relevant information.

    Uses AST parsing to accurately extract:
    - Module structure
    - Class definitions
    - Function/method signatures
    - Type hints
    - Docstrings
    - Decorators
    - Inheritance relationships
    """

    def __init__(self):
        """Initialize the code analyzer."""
        self.elements: List[CodeElement] = []
        self.current_module: Optional[CodeElement] = None
        self.current_class: Optional[CodeElement] = None

    def analyze_file(self, file_path: str) -> List[CodeElement]:
        """
        Analyze a Python file and extract code elements.

        Args:
            file_path: Path to the Python file

        Returns:
            List of CodeElement objects
        """
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        return self.analyze_content(content, file_path)

    def analyze_content(self, content: str, file_path: str = None) -> List[CodeElement]:
        """
        Analyze Python code content and extract code elements.

        Args:
            content: Python code content
            file_path: Optional file path

        Returns:
            List of CodeElement objects
        """
        self.elements = []
        self.current_module = None
        self.current_class = None

        try:
            tree = ast.parse(content)

            # Create module element
            module_docstring = ast.get_docstring(tree)
            self.current_module = CodeElement(
                name=Path(file_path).stem if file_path else "module",
                element_type=ElementType.MODULE,
                docstring=module_docstring,
                file_path=file_path,
                line_number=1,
            )
            self.elements.append(self.current_module)

            # Visit AST nodes
            visitor = DocumentationVisitor(self)
            visitor.visit(tree)

        except SyntaxError as e:
            print(f"Syntax error in {file_path}: {e}")

        return self.elements

    def analyze_directory(
        self, directory_path: str, patterns: List[str] = None
    ) -> List[CodeElement]:
        """
        Analyze all Python files in a directory.

        Args:
            directory_path: Path to the directory
            patterns: File patterns to match (default: ["*.py"])

        Returns:
            List of all CodeElement objects
        """
        if patterns is None:
            patterns = ["*.py"]

        all_elements = []
        directory = Path(directory_path)

        for pattern in patterns:
            for file_path in directory.rglob(pattern):
                if file_path.is_file():
                    elements = self.analyze_file(str(file_path))
                    all_elements.extend(elements)

        return all_elements

    def extract_api_endpoints(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Extract FastAPI endpoint information from a file.

        Args:
            file_path: Path to the file containing FastAPI routes

        Returns:
            List of endpoint information dictionaries
        """
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        endpoints = []
        tree = ast.parse(content)

        class FastAPIVisitor(ast.NodeVisitor):
            def __init__(self, endpoints_list):
                self.endpoints = endpoints_list

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                # Check for FastAPI route decorators
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Call):
                        if isinstance(decorator.func, ast.Attribute):
                            if decorator.func.attr in [
                                "get",
                                "post",
                                "put",
                                "delete",
                                "patch",
                                "options",
                                "head",
                            ]:
                                endpoint_info = {
                                    "name": node.name,
                                    "method": decorator.func.attr.upper(),
                                    "path": self._extract_path(decorator),
                                    "docstring": ast.get_docstring(node),
                                    "line_number": node.lineno,
                                    "parameters": self._extract_parameters(node),
                                    "returns": self._extract_return_type(node),
                                }
                                self.endpoints.append(endpoint_info)
                self.generic_visit(node)

            def _extract_path(self, decorator: ast.Call) -> str:
                """Extract path from decorator."""
                if decorator.args:
                    if isinstance(decorator.args[0], ast.Str):
                        return decorator.args[0].s
                    elif isinstance(decorator.args[0], ast.Constant):
                        return str(decorator.args[0].value)
                return "/"

            def _extract_parameters(self, node: ast.FunctionDef) -> List[Dict[str, str]]:
                """Extract function parameters."""
                params = []
                for arg in node.args.args:
                    param_info = {"name": arg.arg, "type": None}
                    if arg.annotation:
                        param_info["type"] = ast.unparse(arg.annotation)
                    params.append(param_info)
                return params

            def _extract_return_type(self, node: ast.FunctionDef) -> Optional[str]:
                """Extract return type."""
                if node.returns:
                    return ast.unparse(node.returns)
                return None

        visitor = FastAPIVisitor(endpoints)
        visitor.visit(tree)

        return endpoints

    def extract_examples_from_tests(
        self, test_file_path: str, source_file_path: str
    ) -> List[str]:
        """
        Extract code examples from test files.

        Args:
            test_file_path: Path to the test file
            source_file_path: Path to the corresponding source file

        Returns:
            List of example code snippets
        """
        examples = []

        try:
            with open(test_file_path, "r", encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)

            class TestExampleVisitor(ast.NodeVisitor):
                def __init__(self, examples_list):
                    self.examples = examples_list

                def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                    if node.name.startswith("test_"):
                        # Extract function body as example
                        example_code = ast.unparse(node)
                        self.examples.append(example_code)
                    self.generic_visit(node)

            visitor = TestExampleVisitor(examples)
            visitor.visit(tree)

        except Exception as e:
            print(f"Error extracting examples from {test_file_path}: {e}")

        return examples

    def get_import_structure(self, file_path: str) -> Dict[str, List[str]]:
        """
        Extract import structure from a file.

        Args:
            file_path: Path to the file

        Returns:
            Dictionary with 'standard', 'third_party', and 'local' imports
        """
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        imports = {"standard": [], "third_party": [], "local": []}
        tree = ast.parse(content)

        class ImportVisitor(ast.NodeVisitor):
            def __init__(self, imports_dict):
                self.imports = imports_dict

            def visit_Import(self, node: ast.Import) -> None:
                for alias in node.names:
                    module_name = alias.name.split(".")[0]
                    self._categorize_import(module_name)
                self.generic_visit(node)

            def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
                if node.module:
                    module_name = node.module.split(".")[0]
                    self._categorize_import(module_name)
                self.generic_visit(node)

            def _categorize_import(self, module_name: str) -> None:
                """Categorize import by type."""
                standard_libs = {
                    "os",
                    "sys",
                    "re",
                    "json",
                    "datetime",
                    "typing",
                    "pathlib",
                    "collections",
                    "itertools",
                    "functools",
                    "math",
                    "random",
                }
                if module_name in standard_libs:
                    self.imports["standard"].append(module_name)
                elif module_name.startswith("astroml"):
                    self.imports["local"].append(module_name)
                else:
                    self.imports["third_party"].append(module_name)

        visitor = ImportVisitor(imports)
        visitor.visit(tree)

        return imports


class DocumentationVisitor(ast.NodeVisitor):
    """AST visitor for extracting documentation elements."""

    def __init__(self, analyzer: CodeAnalyzer):
        self.analyzer = analyzer

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit class definition."""
        class_element = CodeElement(
            name=node.name,
            element_type=ElementType.CLASS,
            docstring=ast.get_docstring(node),
            file_path=self.analyzer.current_module.file_path
            if self.analyzer.current_module
            else None,
            line_number=node.lineno,
            decorators=[ast.unparse(d) for d in node.decorator_list],
            metadata={
                "bases": [ast.unparse(base) for base in node.bases],
            },
        )

        # Extract type hints from class variables
        for stmt in node.body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                class_element.type_hints[stmt.target.id] = ast.unparse(stmt.annotation)

        if self.analyzer.current_module:
            class_element.parent = self.analyzer.current_module
            self.analyzer.current_module.children.append(class_element)

        # Set as current class and visit children
        previous_class = self.analyzer.current_class
        self.analyzer.current_class = class_element
        self.analyzer.elements.append(class_element)

        self.generic_visit(node)

        self.analyzer.current_class = previous_class

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definition."""
        element_type = (
            ElementType.METHOD
            if self.analyzer.current_class
            else ElementType.FUNCTION
        )

        # Extract parameters
        parameters = []
        for arg in node.args.args:
            param_info = {"name": arg.arg}
            if arg.annotation:
                param_info["type"] = ast.unparse(arg.annotation)
            if arg.arg in node.args.defaults:
                param_info["default"] = ast.unparse(
                    node.args.defaults[node.args.args.index(arg)]
                )
            parameters.append(param_info)

        # Extract return type
        returns = ast.unparse(node.returns) if node.returns else None

        # Extract examples from docstring
        docstring = ast.get_docstring(node)
        examples = []
        if docstring:
            examples = self._extract_examples(docstring)

        function_element = CodeElement(
            name=node.name,
            element_type=element_type,
            docstring=docstring,
            file_path=self.analyzer.current_module.file_path
            if self.analyzer.current_module
            else None,
            line_number=node.lineno,
            signature=ast.unparse(node),
            decorators=[ast.unparse(d) for d in node.decorator_list],
            parameters=parameters,
            returns=returns,
            examples=examples,
            metadata={
                "is_async": isinstance(node, ast.AsyncFunctionDef),
                "is_property": any(
                    "property" in ast.unparse(d) for d in node.decorator_list
                ),
            },
        )

        # Extract type hints
        for param in parameters:
            if param.get("type"):
                function_element.type_hints[param["name"]] = param["type"]
        if returns:
            function_element.type_hints["return"] = returns

        if self.analyzer.current_class:
            function_element.parent = self.analyzer.current_class
            self.analyzer.current_class.children.append(function_element)
        elif self.analyzer.current_module:
            function_element.parent = self.analyzer.current_module
            self.analyzer.current_module.children.append(function_element)

        self.analyzer.elements.append(function_element)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Visit async function definition."""
        self.visit_FunctionDef(node)

    def _extract_examples(self, docstring: str) -> List[str]:
        """Extract code examples from docstring."""
        examples = []
        lines = docstring.split("\n")
        in_example = False
        example_lines = []

        for line in lines:
            if ">>>" in line or "Example:" in line:
                in_example = True
                example_lines.append(line)
            elif in_example:
                if line.strip() and not line.startswith(" "):
                    in_example = False
                    if example_lines:
                        examples.append("\n".join(example_lines))
                        example_lines = []
                else:
                    example_lines.append(line)

        if example_lines:
            examples.append("\n".join(example_lines))

        return examples
