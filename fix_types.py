import os
import re

def add_type_hints(filepath):
    with open(filepath, "r") as f:
        content = f.read()
    
    if "def " not in content:
        return
        
    lines = content.split('\n')
    modified = False
    needs_any = False
    
    for i, line in enumerate(lines):
        if line.strip().startswith("def "):
            # if no return type
            if "->" not in line and ":" in line:
                # Add -> Any
                lines[i] = line.replace(":", " -> Any:", 1)
                modified = True
                needs_any = True
                
    if modified:
        content = '\n'.join(lines)
        if needs_any and "from typing import" not in content and "import typing" not in content:
            content = "from typing import Any, Dict, List, Optional, Union, Callable\n" + content
        with open(filepath, "w") as f:
            f.write(content)

for root, dirs, files in os.walk("astroml"):
    for file in files:
        if file.endswith(".py"):
            add_type_hints(os.path.join(root, file))
