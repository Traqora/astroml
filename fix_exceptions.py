import os
import re

exception_hierarchy = """class AstroMLError(Exception):
    def __init__(self, message: str, **context):
        super().__init__(message)
        self.context = context

class IngestionError(AstroMLError): pass
class FeatureError(AstroMLError): pass
class ModelError(AstroMLError): pass
class DatabaseError(AstroMLError): pass
"""

with open("astroml/utils/exceptions.py", "w") as f:
    f.write(exception_hierarchy)

for root, dirs, files in os.walk("astroml"):
    for file in files:
        if file.endswith(".py") and file != "exceptions.py":
            path = os.path.join(root, file)
            with open(path, "r") as f:
                content = f.read()
            
            if "except Exception" in content:
                # Add import if needed
                if "AstroMLError" not in content:
                    content = "from astroml.utils.exceptions import AstroMLError\n" + content
                
                # Replace
                content = content.replace("except Exception as e:", "except AstroMLError as e:")
                content = content.replace("except Exception:", "except AstroMLError:")
                
                with open(path, "w") as f:
                    f.write(content)
