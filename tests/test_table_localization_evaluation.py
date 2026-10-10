import json
from tabulus.evaluation import evaluate_table_localization

def test_localization(tmp_path):
    g = tmp_path/"g.json"; p = tmp_path/"p.json"
    g.write_text(json.dumps({"tables":[
        {"table_id":"page_001_table_001","page":1,"annotation":{"image_name":"page_001_table_001.png"}},
        {"table_id":"page_002_table_002","page":2,"annotation":{"image_name":"legacy.png"}},
    ]}))
    p.write_text(json.dumps({"tables":[
        {"table_id":1,"page_nr":1,"image_name":"page_001_table_001.png"},
        {"table_id":7,"page_nr":2,"image_name":"different.png"},
        {"table_id":8,"page_nr":3,"image_name":"extra.png"},
    ]}))
    r = evaluate_table_localization(g,p)
    assert (r.matched_fragments,r.false_positives,r.false_negatives)==(2,1,0)
    assert (r.matched_by_identity,r.matched_by_page_order)==(1,1)
