import sys
sys.path.append('/home/svu/e1352224/workspace/kv-cache/spotedit')

print("=" * 60)
print("Import Diagnostic")
print("=" * 60)

# 1. Check if module exists
try:
    import Qwen_image_edit
    print("✓ Qwen_image_edit module found")
    print(f"  Location: {Qwen_image_edit.__file__}")
except ImportError as e:
    print(f"✗ Cannot import Qwen_image_edit: {e}")
    sys.exit(1)

# 2. Check __init__.py contents
print("\n" + "=" * 60)
print("Checking __init__.py contents")
print("=" * 60)
import Qwen_image_edit
print(f"Available attributes: {dir(Qwen_image_edit)}")

# 3. Try importing submodules
print("\n" + "=" * 60)
print("Testing submodule imports")
print("=" * 60)

try:
    from Qwen_image_edit.qwen_spot_ultis import SpotEditConfig
    print("✓ SpotEditConfig imported")
except ImportError as e:
    print(f"✗ SpotEditConfig import failed: {e}")

try:
    from Qwen_image_edit.qwen_spotedit import generate
    print("✓ generate imported")
except ImportError as e:
    print(f"✗ generate import failed: {e}")
    import traceback
    traceback.print_exc()

# 4. Final test
print("\n" + "=" * 60)
print("Final import test")
print("=" * 60)

try:
    from Qwen_image_edit import generate, SpotEditConfig
    print("✓ SUCCESS: All imports working!")
except ImportError as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()