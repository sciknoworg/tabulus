import json
from tabulus.evaluation import evaluate_reference_resolution

def test_step6(tmp_path):
    g=tmp_path/"g.json"; p=tmp_path/"p.json"
    g.write_text(json.dumps({"records":[
        {"step4_index":1,"gold_doi":"10.1000/A","gold_title":"A Scientific Paper",
         "gold_authors":["Jane Smith","John Doe"],"gold_year":2020,
         "gold_venue":"Journal of Vacuum Science and Technology A"},
        {"step4_index":2,"gold_doi":"","gold_title":"Second Paper",
         "gold_authors":["Alice Brown"],"gold_year":2021,
         "gold_venue":"Applied Physics Letters"},
        {"step4_index":3,"gold_doi":None,"gold_title":"Third Paper",
         "gold_authors":["Bob Green"],"gold_year":2022,"gold_venue":"Nature"},
    ]}))
    p.write_text(json.dumps({"entries":[
        {"resolution":{"reference_index":1,"status":"validated_with_doi",
         "canonical_doi":"https://doi.org/10.1000/a","canonical_title":"A Scientific Paper",
         "canonical_authors":["Jane Smith","John Doe"],"canonical_year":2020,
         "canonical_venue":"J. Vacuum Sci. Technol. A"}},
        {"resolution":{"reference_index":2,"status":"validated_without_doi",
         "canonical_doi":"","canonical_title":"Second Paper","canonical_authors":["Alice Brown"],
         "canonical_year":2021,"canonical_venue":"Applied Physics Letters"}},
        {"resolution":{"reference_index":3,"status":"rejected","canonical_doi":"",
         "canonical_title":"","canonical_authors":[],"canonical_year":None,"canonical_venue":""}},
    ]}))
    r=evaluate_reference_resolution(g,p)
    assert r.resolution_yield==2/3
    assert (r.doi_evaluable,r.doi_predictions,r.doi_exact_matches)==(1,1,1)
    assert r.doi_f1==1.0
    assert (r.title_correct,r.authors_correct,r.year_correct,r.venue_correct)==(2,2,2,2)


def test_step6_uses_gold_index_and_accepts_string_gold_authors(tmp_path):
    g=tmp_path/"g2.json"; p=tmp_path/"p2.json"
    g.write_text(json.dumps({"records":[{
        "gold_index":7,
        "step4_index":99,
        "gold_doi":"",
        "gold_title":"Example Scientific Article",
        "gold_authors":"Jane Smith and John Doe",
        "gold_year":"2020",
        "gold_venue":"Appl. Surf. Sci."
    }]}))
    p.write_text(json.dumps({"entries":[{
        "resolution":{
            "reference_index":7,
            "status":"validated_without_doi",
            "canonical_doi":"",
            "canonical_title":"Example Scientific Article",
            "canonical_authors":["Jane Smith","John Doe"],
            "canonical_year":2020,
            "canonical_venue":"Applied Surface Science"
        }
    }]}))
    r=evaluate_reference_resolution(g,p)
    assert r.matched_predictions==1
    assert r.authors_correct==1
    assert r.title_correct==1
    assert r.year_correct==1
    assert r.venue_correct==1
