import sys
import importlib

if len(sys.argv) < 2:
    print("Usage: python -m protconv <script_name> [args...]")
    sys.exit(1)

script_name = sys.argv[1]
try:
    module = importlib.import_module(f"protconv.scripts.{script_name}")
except ImportError as e:
    print(f"Could not import script 'protconv.scripts.{script_name}': {e}")
    sys.exit(1)

if not hasattr(module, "main"):
    print(f"Script 'protconv.scripts.{script_name}' does not have a main() function.")
    sys.exit(1)

# Pass remaining args to the script's main()
module.main(*sys.argv[2:])
