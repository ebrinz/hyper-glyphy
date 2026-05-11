import importlib.util
from pathlib import Path

_MOD_PATH = Path(__file__).resolve().parents[1] / "scripts" / "01_parse_tlhdig.py"


def _load():
    spec = importlib.util.spec_from_file_location("parse_tlhdig", _MOD_PATH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


SAMPLE_XML = """<?xml version="1.0"?><AOxml xmlns:hpm="http://hethiter.net/ns/hpm/1.0" xml:space="preserve" >
<AOHeader><docID>KBo 49.259</docID></AOHeader>
<body><div1 type="transliteration"><text xml:lang="Hit">
<lb txtid="KBo 49.259" lnr="Vs. 1" lg="Hit" cu="X"/>
<w trans="nuššan" mrp0sel=" " mrp1="nu=ššan@@ CONNn=OBPs@@ ">nu-uš-ša-an</w>
<w trans="atanzi" mrp0sel=" 1 " mrp1="ēd-/ad-@essen@3PL.PRS@I.2.4@">a-da-an-zi</w>
<lb txtid="KBo 49.259" lnr="Vs. 2" lg="Hit" cu="X"/>
<w trans="É" mrp0sel=" " mrp1="É@Haus@{ a → NOM.SG(UNM)}@28.16@"><sGr>É</sGr></w>
<w trans="DINGIR-LIM" mrp0sel=" " mrp1="DINGIR-LIM@Gott@{ a → NOM.SG(UNM)}@28.3.1.1.1@"><sGr>DINGIR</sGr><aGr>-LIM</aGr></w>
</text></div1></body></AOxml>"""


def test_parse_mrp_field_simple():
    mod = _load()
    parsed = mod.parse_mrp("nu=ššan@@ CONNn=OBPs@@ ")
    assert parsed["lemma"] == "nu=ššan"
    assert parsed["gloss"] == ""
    assert "CONN" in parsed["features"]


def test_parse_mrp_field_with_gloss():
    mod = _load()
    parsed = mod.parse_mrp("DINGIR-LIM@Gott@{ a → NOM.SG(UNM)}@28.3.1.1.1@")
    assert parsed["lemma"] == "DINGIR-LIM"
    assert parsed["gloss"] == "Gott"
    assert "NOM.SG" in parsed["features"]
    assert parsed["category"] == "28.3.1.1.1"


def test_parse_mrp_empty_input():
    mod = _load()
    assert mod.parse_mrp("") is None
    assert mod.parse_mrp(None) is None


def test_select_mrp_uses_mrp0sel():
    """mrp0sel like ' 2 ' selects mrp2; ' ' (default empty) picks mrp1."""
    mod = _load()
    attrs = {
        "mrp0sel": " 2 ",
        "mrp1": "form1@gloss1@feat1@cat1@",
        "mrp2": "form2@gloss2@feat2@cat2@",
        "mrp3": "form3@gloss3@feat3@cat3@",
    }
    selected = mod.select_mrp(attrs)
    assert selected["lemma"] == "form2"

    # Default (empty mrp0sel) picks mrp1
    attrs["mrp0sel"] = " "
    selected = mod.select_mrp(attrs)
    assert selected["lemma"] == "form1"

    # DEL means dropped/uncertain — return None
    attrs["mrp0sel"] = "DEL"
    assert mod.select_mrp(attrs) is None


def test_extract_words_from_xml():
    mod = _load()
    parsed = mod.parse_xml_string(SAMPLE_XML, doc_id="KBo 49.259")
    assert parsed["doc_id"] == "KBo 49.259"
    # 4 words in the sample
    words = parsed["words"]
    assert len(words) == 4
    # First word: trans=nuššan, mrp1 lemma=nu=ššan
    assert words[0]["trans"] == "nuššan"
    assert words[0]["mrp"]["lemma"] == "nu=ššan"
    # Fourth word: heterogram with both Sumerogram and Akkadogram parts
    assert words[3]["trans"] == "DINGIR-LIM"
    assert words[3]["has_sumerogram"] is True
    assert words[3]["has_akkadogram"] is True
    assert "DINGIR" in words[3]["sumerograms"]
    assert "LIM" in words[3]["akkadograms"][0] or words[3]["akkadograms"][0] == "-LIM"


def test_extract_lines():
    mod = _load()
    parsed = mod.parse_xml_string(SAMPLE_XML, doc_id="KBo 49.259")
    # Two <lb> markers split the document into 2 lines
    lines = parsed["lines"]
    assert len(lines) == 2
    assert "nuššan" in lines[0]
    assert "atanzi" in lines[0]
    assert "DINGIR-LIM" in lines[1]
