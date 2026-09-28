import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from elderly_monitor.environment import load_project_env


class EnvironmentTests(unittest.TestCase):
    def load(self, content):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / '.env'
            path.write_text(content, encoding='utf-8')
            load_project_env(path)

    @patch.dict(os.environ, {}, clear=True)
    def test_loads_quoted_key_and_ignores_other_variables(self):
        self.load('# comment\nGEMINI_API_KEY="example-test-key" # comment\nOTHER_SECRET=x')
        self.assertEqual(os.environ.get('GEMINI_API_KEY'), 'example-test-key')
        self.assertNotIn('OTHER_SECRET', os.environ)

    @patch.dict(os.environ, {'GEMINI_API_KEY':'shell-value'}, clear=True)
    def test_existing_environment_wins(self):
        self.load('GEMINI_API_KEY=file-value')
        self.assertEqual(os.environ['GEMINI_API_KEY'],'shell-value')

    @patch.dict(os.environ, {}, clear=True)
    def test_blank_template_does_not_set_key(self):
        self.load('GEMINI_API_KEY= # fill locally')
        self.assertNotIn('GEMINI_API_KEY',os.environ)

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_file_is_optional(self):
        with tempfile.TemporaryDirectory() as directory:
            load_project_env(Path(directory)/'.env')
        self.assertNotIn('GEMINI_API_KEY',os.environ)
