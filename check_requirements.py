import pkg_resources

print("Checking installed packages against requirements.txt...")

try:
    with open('requirements.txt') as f:
        requirements = f.read().splitlines()
except FileNotFoundError:
    print("❌ requirements.txt not found.")
    exit(1)

if not requirements:
    print("⚠️ requirements.txt is empty.")
    exit(0)

for req in requirements:
    req = req.strip()
    if not req or req.startswith("#"):
        continue  # skip empty lines or comments
    try:
        pkg_resources.require(req)
        print(f"✅ {req} is installed and matches version")
    except pkg_resources.DistributionNotFound:
        print(f"❌ {req} is NOT installed")
    except pkg_resources.VersionConflict as e:
        print(f"⚠️ Version conflict for {req}: {e.report()}")
