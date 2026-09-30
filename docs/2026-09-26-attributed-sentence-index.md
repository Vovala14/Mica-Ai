# Tatoeba source-linked sentence index (2026-09-26)

## Why this exists

The internal retrieval indexes built from `r1/data/everyday/train.txt` have no
record-level source URL, title, or license. Those indexes should not be used
as-is for a public example site. `r1/data/build_attributed_tatoeba_index.py`
builds a separate, compatible index from the locally downloaded Tatoeba
English export. It retains only sentences that occur exactly in `train.txt`,
whose Tatoeba sentence ID hashes into the train split, and whose normalized
text is absent from `val.txt` and `test.txt`.

The builder joins the official detailed English export by sentence ID and
**exact text**. A later export can revise/remove sentences; those rows are
discarded. The detailed export supplies the current *owner* username, which
is kept in `attribution.jsonl`. Owner is not necessarily the author, so
`source_author` remains null. The retrieval decoder receives source sentence
URL/title/license through its existing `Sentence` fields; it does not receive
the owner sidecar.

## Sources and terms checked

- [Tatoeba's downloads page](https://tatoeba.org/en/downloads) describes
  the basic export as ID, language, and text; the detailed export adds owner
  name and dates. It says export files are under CC BY 2.0 FR, and some
  sentences are also available under CC0 1.0.
- [Tatoeba's current FAQ](https://en.wiki.tatoeba.org/articles/show/faq)
  instructs users of the textual data to credit Tatoeba with a link and
  mention CC BY 2.0 FR. Its guidance for listing individual contributors
  concerns audio.
- [Tatoeba's archived older terms](https://en.wiki.tatoeba.org/articles/show/terms-of-use-v1)
  permitted linking to each sentence page as author attribution and called
  for a change notice and license link. This is historical guidance, not the
  current policy source.

The local COCO annotation JSON attaches licenses to *images*, not captions;
FineWeb Parquet has source URLs but no source rights field; SimpleWiki Parquet
has URLs/titles but no license field. These were excluded from this
source-linked index. The training model can still have seen these sources;
the claim here concerns only the displayed retrieval candidates.

## Built artifacts

The canonical source-linked index is
`r1/data/everyday/attributed_tatoeba_index_1m_detailed/`:

- 826,954 source-linked, train-only sentences, from a 1,000,000-sentence cap.
  This is the entire eligible Tatoeba pool under the filter; the cap cannot
  produce a full million.
- The detailed export matched ID and text exactly for all retained rows. Two
  changed/missing source rows were discarded. 26,150 retained rows have no
  owner username in the detailed export.
- `sentences.jsonl` has direct source URL, generic title, and the export-level
  license label for every row. `attribution.jsonl` keeps the source ID,
  owner, source-text hash, and license evidence. `manifest.json` records SHA-256
  hashes for the index, both Tatoeba exports, and all three everyday splits.
- `source_provenance_ready=true` describes the completed source join.
  `publication_ready=false` remains until the example site's actual display
  includes the required attribution and a visible-result audit is done.

The downloaded detailed export is 34,909,145 compressed bytes,
SHA-256 `353d48de7905952cf6f1500f6a3158516ecf9e10cd844ba051982cfa4a11c111`.
Its official URL is
`https://downloads.tatoeba.org/exports/per_language/eng/eng_sentences_detailed.tsv.bz2`.

| Exact file | SHA-256 |
| --- | --- |
| `r1/data/external/tatoeba/eng_sentences.tsv.bz2` | `69a8ff3de808313a75c5305f4f7b3f4c9d3460c1bd9fdf0f375da247c2d02489` |
| `r1/data/external/tatoeba/eng_sentences_detailed.tsv.bz2` | `353d48de7905952cf6f1500f6a3158516ecf9e10cd844ba051982cfa4a11c111` |
| `r1/data/everyday/train.txt` | `af4af14e6bd44e9647dad16a0d1db5db040a67393b6abb0e092a5d348ad71bc6` |
| `r1/data/everyday/val.txt` | `09c08a6e0077b712b584ed49231744e51aad077f77d3aeaa7f919fdddf4375db` |
| `r1/data/everyday/test.txt` | `34f6e90387ce5f93251d94824346a061c2792bccbbcc175becedb84b143c744f` |
| `r1/data/everyday/attributed_tatoeba_index_1m_detailed/sentences.jsonl` | `867d5aa671b5edf66542587bea3a198912baa26344ed9fe33a8f62669ae56dfa` |
| `r1/data/everyday/attributed_tatoeba_index_1m_detailed/attribution.jsonl` | `de7efbd0537aa7fb3ddb1b1e0e2b4e55b29c888f8e492dfe48685922bafb9fb3` |

To rebuild with the same local inputs:

```powershell
& 'C:\Users\vlavrik\PycharmProjects\.venv-rocm\Scripts\python.exe' r1\data\build_attributed_tatoeba_index.py `
  --train r1\data\everyday\train.txt `
  --tatoeba r1\data\external\tatoeba\eng_sentences.tsv.bz2 `
  --detailed r1\data\external\tatoeba\eng_sentences_detailed.tsv.bz2 `
  --out r1\data\everyday\attributed_tatoeba_index_1m_detailed `
  --max-sentences 1000000
```

The builder refuses a nonempty output directory. Rebuild into a new directory.

## Decoder development check

Using the current retrieval decoder and the 20 **development** prompts, the
826,954-sentence index returned a suggestion on 18/20 sentence prompts and
19/20 short prompts. These are coverage counts, not a coherence pass. Examples:

| Prompt | Sentence-mode output | Source |
| --- | --- | --- |
| `I need to buy some milk and` | `I need to buy some milk and eggs.` | [#10946971](https://tatoeba.org/en/sentences/show/10946971) |
| `It was raining so hard that` | `It was raining so hard that we had to put off our departure.` | [#26996](https://tatoeba.org/en/sentences/show/26996) |
| `Before you leave, please turn off the` | `Before you leave, please turn off the lights in the kitchen.` | [#9795939](https://tatoeba.org/en/sentences/show/9795939) |
| `Please close the` | `Please close the door.` | [#12463943](https://tatoeba.org/en/sentences/show/12463943) |

It abstained on `The meeting has been moved to` and `The kids are finally
asleep, so` in sentence mode. `I will send you the report by the end of play.`
is a grammatically valid but awkward topic choice. The sealed sentence-quality
holdout was not inspected or used here.

## Conditions for a public demo

The interface should say that suggestions are adapted from Tatoeba sentences
and ranked by MICA, link each suggestion to its sentence page, credit
[Tatoeba](https://tatoeba.org), and link [CC BY 2.0 FR](https://creativecommons.org/licenses/by/2.0/fr/).
These conditions reflect the publisher's current FAQ and export notice. The
source-linked index provides the data needed for this display; it cannot
verify that an interface actually presents it. A separate visible-result
review should catch awkward suggestions, quoted material, and missing source
pages before publishing.
