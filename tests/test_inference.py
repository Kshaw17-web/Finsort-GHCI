import pytest
from finsort.inference import predict_category


def test_predict_category_returns_dict():
    """Test that predict_category returns a dict with all required keys."""
    # This test requires model.pkl and vectorizer.pkl to exist
    result = predict_category('SQ *COFFEE-SPOT 123')
    
    assert isinstance(result, dict), "Result should be a dictionary"
    required_keys = {'raw', 'cleaned', 'tag', 'category', 'confidence'}
    assert set(result.keys()) == required_keys, f"Result should have keys {required_keys}, got {set(result.keys())}"


def test_predict_category_key_types():
    """Test that each key in the result has the correct type."""
    result = predict_category('AMAZON MKTPLACE PMTS')
    
    assert isinstance(result['raw'], str), "'raw' should be a string"
    assert isinstance(result['cleaned'], str), "'cleaned' should be a string"
    assert isinstance(result['tag'], str), "'tag' should be a string"
    assert isinstance(result['category'], str), "'category' should be a string"
    assert isinstance(result['confidence'], (int, float)), "'confidence' should be a number"


def test_predict_category_values():
    """Test that predict_category returns meaningful values."""
    result = predict_category('SQ *COFFEE-SPOT 123')
    
    # raw should match input
    assert result['raw'] == 'SQ *COFFEE-SPOT 123'
    
    # cleaned should be non-empty
    assert len(result['cleaned']) > 0
    
    # tag should be non-empty
    assert len(result['tag']) > 0
    
    # category should be non-empty
    assert len(result['category']) > 0
    
    # confidence should be between 0 and 1
    assert 0.0 <= result['confidence'] <= 1.0
