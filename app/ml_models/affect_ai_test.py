import affect_ai
import pytest

# words: foo, bar, baz, goo, car, caz, hoo, dar, daz, ioo, ear, eaz, loo, far, faz; corpora: happiness 1, satisfaction 2, elation 2, 3

# Test that an affect_AI object gets created correctly
def test_creation():
    # We create an affect_ai object with some parameters
    # We make sure those parameters do what they should within the object
    pass
# Test that an affect_AI object can be trained, and builds vocabulary correctly
def test_training():
    # We try to pass in corpora to the affect_ai object we created earlier
    # We make sure its internal objects change as they should
    pass
# Test that an affect_AI object correctly scores samples
def test_scoring():
    # We have the affect_ai score a sample of words containing some of its trained words
    # We compare the scored result to what we know it should be
    pass
