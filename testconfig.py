# test_config_structure.py
try:
    from config.settings import config
    
    print("✅ Configuration Structure Test:")
    print(f"  - Has llm_config: {hasattr(config, 'llm_config')}")
    print(f"  - llm_config keys: {list(config.llm_config.keys()) if hasattr(config, 'llm_config') else 'None'}")
    
    if hasattr(config, 'llm_config'):
        providers = config.llm_config.get('providers', {})
        print(f"  - Providers in llm_config: {list(providers.keys())}")
        
        for prov_name, prov_data in providers.items():
            has_api_key = 'api_key' in prov_data
            print(f"    - {prov_name}: has_api_key={has_api_key}")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
