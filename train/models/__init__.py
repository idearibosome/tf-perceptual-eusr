import glob
import importlib
import os
import re

__module_file_regexp = r'(.+)\.py(c?)$'
__current_path = os.path.dirname(os.path.realpath(__file__))

for entry in os.listdir(__current_path):
  if (entry == '__init__.py'):
    continue
  if (os.path.isfile(os.path.join(__current_path, entry))):
    regexp_result = re.search(__module_file_regexp, entry)
    if regexp_result:
      importlib.import_module('models.'+regexp_result.groups()[0])