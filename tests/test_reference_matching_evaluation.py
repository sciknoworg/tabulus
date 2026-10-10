import json
from tabulus.evaluation import evaluate_reference_matching

def test_step5(tmp_path):
    g=tmp_path/"g.json"; p=tmp_path/"p.json"
    g.write_text(json.dumps({"tables":[{
        "table_id":"page_001_table_001","reference_column_index":2,
        "citation_cells":[
            {"value":"12, 14","bibliography_indices":[12,14]},
            {"value":"20--22","bibliography_indices":[20,21,22]},
        ]}]}))
    p.write_text(json.dumps({"matched_tables":[{
        "source_prediction":"predictions/page_001_table_001.csv","reference_column_index":2,
        "matches":[
            {"value":"Refs.","is_header":True,"matched_reference_indices":[]},
            {"value":"[12, 14]","is_header":False,"matched_reference_indices":[12,14]},
            {"value":"20, 21, 22","is_header":False,"matched_reference_indices":[20,21,22]},
        ]}]}))
    r=evaluate_reference_matching(g,p)
    assert r.exact_cell_accuracy==1.0
    assert (r.link_true_positives,r.link_false_positives,r.link_false_negatives)==(5,0,0)
    assert r.link_f1==1.0


def test_step5_historical_controlled_artifact_without_cell_values(tmp_path):
    g=tmp_path/"g2.json"; p=tmp_path/"p2.json"
    g.write_text(json.dumps({"tables":[{
        "table_id":"page_001_table_001",
        "reference_column_index":2,
        "citation_cells":[
            {"value":"44","bibliography_indices":[44]},
            {"value":"45--46","bibliography_indices":[45,46]},
        ]}]}))
    p.write_text(json.dumps({"matched_tables":[{
        "table_id":"page_001_table_001",
        "matches":[
            {"matched_reference_indices":[44]},
            {"matched_reference_indices":[45,46]},
        ]}]}))
    r=evaluate_reference_matching(g,p)
    assert r.exact_link_set_cells==2
    assert r.gold_citation_cells==2
    assert r.link_f1==1.0
