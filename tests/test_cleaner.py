from finsort.cleaner import clean_transaction


def test_clean_empty_string():
    """Test that empty string returns empty string."""
    assert clean_transaction('') == ''
    assert clean_transaction('   ') == ''


def test_clean_typical_messy_inputs():
    """Test typical messy transaction inputs."""
    # Basic cleaning
    assert clean_transaction('SQ *STARBUCKS 123').startswith('starbucks')
    assert clean_transaction('AMAZON MKTPLACE PMTS').startswith('amazon marketplace')
    
    # UPI codes removal
    assert 'upi' not in clean_transaction('UPI AMAZON PAY')
    assert 'pmts' not in clean_transaction('PMTS WALMART')
    assert 'trx' not in clean_transaction('TRX WALMART DEL')
    assert 'swipe' not in clean_transaction('SWIPE Best Buy INDIA')
    
    # Non-letter removal (numbers and special chars replaced with spaces)
    assert clean_transaction('ABC123XYZ') == 'abc xyz'
    assert clean_transaction('TEST@#$%^&*()') == 'test'
    
    # Special word normalization
    assert 'marketplace' in clean_transaction('MKTPLACE')
    assert 'department' in clean_transaction('DEPT STORE')
    assert 'company' in clean_transaction('ABC CO')
    
    # Space collapsing
    assert clean_transaction('  MULTIPLE    SPACES  ') == 'multiple spaces'
    
    # Case normalization
    assert clean_transaction('UPPERCASE') == 'uppercase'
    assert clean_transaction('MixedCase') == 'mixedcase'


def test_clean_non_string_input():
    """Test that non-string input returns empty string."""
    assert clean_transaction(None) == ''
    assert clean_transaction(123) == ''
    assert clean_transaction([]) == ''


def test_clean_preserves_words():
    """Test that word replacement doesn't affect other words."""
    # Should not replace 'co' inside 'coffee'
    result = clean_transaction('SQ *COFFEE-SPOT 123')
    assert 'coffee' in result
    assert 'company' not in result
