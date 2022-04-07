import os
import sys
import unittest

root = os.path.join(os.path.dirname(__file__), "..")
if os.path.exists(os.path.join(root, "mycode_front", "_version.py")):
    sys.path.insert(0, os.path.abspath(root))

import mycode_front as my  # noqa: E402


class MyTests(unittest.TestCase):
    def test_has_uncomitted(self):

        self.assertTrue(my.tag.has_uncommitted("4.4.dev1+hash.bash"))
        self.assertFalse(my.tag.has_uncommitted("4.4.dev1+hash"))
        self.assertFalse(my.tag.has_uncommitted("4.4.dev1"))
        self.assertFalse(my.tag.has_uncommitted("4.4"))

    def test_any_has_uncommitted(self):

        m = "main=3.2.1"
        o = "other"

        self.assertTrue(my.tag.any_has_uncommitted([m, f"{o}=4.4.dev1+hash.bash"]))
        self.assertFalse(my.tag.any_has_uncommitted([m, f"{o}=4.4.dev1+hash"]))
        self.assertFalse(my.tag.any_has_uncommitted([m, f"{o}=4.4.dev1"]))
        self.assertFalse(my.tag.any_has_uncommitted([m, f"{o}=4.4"]))

    def test_greater_equal(self):

        self.assertFalse(my.tag.greater_equal("4.4.dev1+hash.bash", "4.4"))
        self.assertFalse(my.tag.greater_equal("4.4.dev1+hash", "4.4"))
        self.assertFalse(my.tag.greater_equal("4.4.dev1", "4.4"))
        self.assertTrue(my.tag.greater_equal("4.4", "4.4"))

    def test_greater(self):

        self.assertFalse(my.tag.greater("4.4.dev1+hash.bash", "4.4"))
        self.assertFalse(my.tag.greater("4.4.dev1+hash", "4.4"))
        self.assertFalse(my.tag.greater("4.4.dev1", "4.4"))
        self.assertFalse(my.tag.greater("4.4", "4.4"))

    def test_less_equal(self):

        self.assertTrue(my.tag.less_equal("4.4.dev1+hash.bash", "4.4"))
        self.assertTrue(my.tag.less_equal("4.4.dev1+hash", "4.4"))
        self.assertTrue(my.tag.less_equal("4.4.dev1", "4.4"))
        self.assertTrue(my.tag.less_equal("4.4", "4.4"))

    def test_less(self):

        self.assertTrue(my.tag.less("4.4.dev1+hash.bash", "4.4"))
        self.assertTrue(my.tag.less("4.4.dev1+hash", "4.4"))
        self.assertTrue(my.tag.less("4.4.dev1", "4.4"))
        self.assertFalse(my.tag.less("4.4", "4.4"))

    def test_all_greater_equal(self):

        a = ["main=3.2.1", "other=4.4"]
        b = ["main=3.2.0", "other=4.4", "more=3.0.0"]
        self.assertTrue(my.tag.all_greater_equal(a, b))

        a = ["main=3.2.1", "other=4.4"]
        b = ["main=3.2.1.dev1", "other=4.4", "more=3.0.0"]
        self.assertTrue(my.tag.all_greater_equal(a, b))

        a = ["main=3.2.1", "other=4.4"]
        b = ["main=3.2.1.dev1+g423e6a8", "other=4.4", "more=3.0.0"]
        self.assertTrue(my.tag.all_greater_equal(a, b))

        a = ["main=3.2.1", "other=4.4"]
        b = ["main=3.2.1.dev1+g423e6a8.d20210902", "other=4.4", "more=3.0.0"]
        self.assertTrue(my.tag.all_greater_equal(a, b))


if __name__ == "__main__":

    unittest.main()
