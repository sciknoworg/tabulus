import json
from tabulus.evaluation import evaluate_reference_table_classification

def test_step3(tmp_path):
    g=tmp_path/"g.json"; p=tmp_path/"p.json"
    g.write_text(json.dumps({"tables":[
        {"table_id":"page_001_table_001","is_reference_table":True},
        {"table_id":"page_002_table_002","is_reference_table":True},
        {"table_id":"page_003_table_003","is_reference_table":False},
        {"table_id":"page_004_table_004","is_reference_table":False},
    ]}))
    p.write_text(json.dumps({"tables":[
        {"source_prediction":"predictions/page_001_table_001.csv","is_reference_table":True},
        {"source_prediction":"predictions/page_002_table_002.csv","is_reference_table":False},
        {"source_prediction":"predictions/page_003_table_003.csv","is_reference_table":True},
        {"source_prediction":"predictions/page_004_table_004.csv","is_reference_table":False},
    ]}))
    r=evaluate_reference_table_classification(g,p)
    assert (r.true_positives,r.false_positives,r.true_negatives,r.false_negatives)==(1,1,1,1)
    assert r.balanced_accuracy==0.5
