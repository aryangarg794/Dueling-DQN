import numpy as np
import pytest
from dueling_dqn.experience_replay.sum_tree import SumTree  

def test_initialization():
    tree_size = 4
    tree = SumTree(size=tree_size)
    
    assert tree.size == tree_size
    assert tree.pointer == 0
    assert isinstance(tree.tree, np.ndarray)
    assert tree.tree.shape[0] == 2 * tree_size - 1
    assert np.all(tree.tree == 0)

def test_add_single():
    tree = SumTree(size=4)
    
    tree.add([1.0])
    
    assert tree.pointer == 1
    assert tree.total == 1

def test_add_multiple():
    tree = SumTree(size=4)
    
    td_errors = [1.0, 2.0, 3.0, 4.0]
    tree.add(td_errors)
    
    assert tree.pointer == 0  
    assert np.isclose(tree.total, sum(td_errors), atol=1e-6)

def test_update_single():
    tree = SumTree(size=4)
    tree.add([1.0, 2.0, 3.0, 4.0])
    
    tree.update([2], [10.0])
    
    expected_total = 1.0 + 2.0 + 10.0 + 4.0
    assert np.isclose(tree.total, expected_total, atol=1e-6)

def test_update_multiple():
    tree = SumTree(size=4)
    tree.add([1.0, 1.0, 1.0, 1.0])
    
    indices = [0, 1, 2]
    new_td_errors = [5.0, 6.0, 7.0]
    tree.update(indices, new_td_errors)
    
    expected_total = 5.0 + 6.0 + 7.0 + 1.0
    assert np.isclose(tree.total, expected_total, atol=1e-6)
    
def test_add_fail():
    tree = SumTree(size=2)
    with pytest.raises(AssertionError) as err: 
        tree.add([1.0, 2.0, 3.0, 4.0])
        
def test_sample_fail():
    tree = SumTree(size=4)
    tree.add([1.0])
    with pytest.raises(AssertionError) as err:
        tree.sample(2)

def test_sample():
    tree = SumTree(size=4)
    tree.add([1.0, 2.0, 3.0, 4.0])
    
    _ = tree.sample(2)
    
    assert False == True

@pytest.fixture
def simple_sumtree():
    tree = SumTree(size=4)

    tree.add([1.0, 2.0, 3.0, 4.0])
    return tree

def test_get_leaf(simple_sumtree):
    tree = simple_sumtree
    
    total = tree.total
    assert np.isclose(total, 10.0)
    
    idx, prio = tree._get(0.5)
    assert prio == pytest.approx(1.0, abs=1e-6)
    
    idx, prio = tree._get(2.5)
    assert prio == pytest.approx(2.0, abs=1e-6)
    
    idx, prio = tree._get(5.5)
    assert prio == pytest.approx(3.0, abs=1e-6)
    
    idx, prio = tree._get(9.0)
    assert prio == pytest.approx(4.0, abs=1e-6)

def test_sample(simple_sumtree):
    tree = simple_sumtree
    batch_size = 3
    
    tree.n_samples = 4  
    
    indices, priorities = tree.sample(batch_size=batch_size)
    
    assert indices.shape == (batch_size,)
    assert priorities.shape == (batch_size,)

    assert np.all(indices >= 0)
    assert np.all(indices >= tree.size)
    
    assert np.all(priorities > 0)


