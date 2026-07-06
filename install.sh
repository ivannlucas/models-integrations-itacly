
#!/bin/bash

# Install core dependencies
pip install -r requirements.txt

# Install detectron2 (IMPORTANT: after torch)
pip install --no-build-isolation \
  'git+https://github.com/facebookresearch/detectron2.git'