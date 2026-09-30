# בחירת כללים ב־MICA — ניסויי 27 בספטמבר 2026

## יעד ושיטת בדיקה

המטרה היא לרדת לכיוון 1 bit/target ולייצר משפטים חדשים והגיוניים באנגלית. כל
הניסויים כאן משתמשים במנוע השלם של **Minimal Inference Cellular Automaton**:
אותם תאים, ערוצים, שלבים, דפי כללים, שש פעולות ניקוד ופעולות VSET. לא נוסף
מודל שפה חיצוני. כל מספרי האימות שלהלן נמדדו מקובצי `.mica` שיוצאו למנוע השלם.

המודל הטוב הקיים, שאומן על מאגר everyday, משיג **1.787362 bits/target** על
1,000 רשומות אימות נקיות. הוא אינו יוצר משפטים טובים בעקביות. לכן אף מודל
מהניסויים האלה לא הועלה לאתר.

## כללים הפיכים לשמירת הקשר

נבנתה בחירה קשיחה בין 243 מצבי כללים המיוצגים בחמישה ערוצי עבודה תלת־ערכיים.
כל דף כלל מעביר את המצב דרך תמורה שלמה, כך שרצף תווים משותף אינו מוחק הבדל
קודם בין שני הקשרים. הערכים של כללי התכונה והקריאה עדיין נלמדים מן הטקסט.

בפיילוט קצר, לאחר סיומת משותפת של 128 תווים, ששת זוגות ההקשרים עדיין קיבלו
התפלגויות שונות; בכל ששת הזוגות של בקרת המצב האקראי הן כבר היו זהות. הדיוק
ב־128 דוגמאות היה 2.571526 לעומת 2.558919 bits/target בבקרה. באימון מלא
של 40,000 רשומות ו־1,500 צעדי התאמה התקבלו **1.840530 bits/target** על
1,000 דוגמאות, פער של ‎+0.053168 לעומת המודל הטוב הקיים. רווח סמך מזווג של
95% לפער: ‎[+0.044780, +0.061452]. משפטים ישירים עדיין חזרו על תבניות
כגון `The doctor and the station of the station...`.

חיפוש בין חמישה מקדמי תמורה הפיכים בחר לפי אימות האימון במקדם 10, אך בדיקת
האימות הנקייה הקצרה נתנה 2.575026 לעומת 2.571526 למקדם 4. שתי שרשראות מצב
העלו את השגיאה ל־2.613895; קריאה ישירה של מצב הכללים העלתה אותה ל־2.579589.
השינויים האלה לא קודמו לאימון ארוך.

## בחירת תנאי כללים מן הטקסט

תוקנה התאמת בניית תבניות הכללים למנוע שבו חלון העבודה הוא תא אחד: בכל 16
השלבים הכלל פועל על התא של התו הנוכחי. תנאי הכללים נבחרים מתוך הקשרים
שכיחים בטקסט האימון, ואז מאוזנת תדירות הפעלת המועמדים בלי לפתוח מחדש
מועמדים שהושבתו. נוסתה גם בחירת הקשרים לפי מידע מנבא על התו הבא.

| ניסוי קצר, אותו תקציב אימון | bits/target נקי, 128 דוגמאות |
|---|---:|
| בחירה אקראית + איזון | 2.730358 |
| תבניות שכיחות בכל השלבים + איזון | 2.715412 |
| תבניות לפי מידע מנבא + איזון | 2.729390 |
| תבניות רק בשלבים 0 ו־8 | 2.744714 |
| תבניות בשלבי הקשר קצר | 2.764310 |
| תבניות רק בשלבים 0–3 | 2.733814 |

השיפור של תבניות שכיחות לעומת בחירה אקראית בפיילוט היה ‎−0.014947
bits/target, עם רווח סמך מזווג של 95% ‎[−0.071844, +0.042050]. הוא אינו
מבוסס מספיק על המדגם הקצר. באימון מלא אותה שיטה הגיעה ל־**1.932266
bits/target** על 1,000 דוגמאות, פער של ‎+0.144905 לעומת המודל הטוב הקיים
(רווח סמך ‎[+0.135451, +0.154623]). גם היצירה חזרה על `with a bathroom`
שוב ושוב. לכן אין להשתמש בבחירת תבניות שכיחות לכל הכללים כגרסת ברירת מחדל.

## מצב והמשך

השמירה על הקשר הוכחה כמנגנון עובד, אך עדיין אינה משפרת את החיזוי או את איכות
המשפטים. בחירת תבניות לפי שכיחות או מידע מנבא לא עמדה בבדיקת אימון מלאה.

## אימון נפרד על פרוזה מסוננת

נבדק מאגר פרוזה אנגלית גדול שסוננו ממנו מתמטיקה, כפילויות וקטעי סימון.
הארכיטקטורה ובחירת הכללים האקראית־מאוזנת של MICA נשמרו; רק טקסט האימון
השתנה. לאחר סבב מלא אחד של 40,000 רשומות ו־1,500 צעדים:

| מערך מוחזק בצד, 1,000 רשומות | מודל everyday הקיים | מודל שאומן על פרוזה |
|---|---:|---:|
| פרוזה | 2.345720 | **2.183297** |
| everyday נקי | **1.787362** | 2.211163 |

על פרוזה השיפור המזווג הוא ‎−0.162423 bits/target, רווח סמך 95%
‎[−0.170291, −0.154685]. על everyday יש הידרדרות של ‎+0.423801,
רווח סמך ‎[+0.393733, +0.453548]. אלו מערכי טקסט שונים, ולכן אין מספר
יחיד שמהווה שיפור כולל. היצירה עדיין אינה מספקת: למשל
`The doctor in the content of the content of the content...` וגם
`The dog and a second the content of the content...`. דגימה אקראית ממיקה
הפחיתה חזרות אך יצרה מילים משובשות ומשפטים לא עקביים. המודל אינו מועמד לאתר.

## הצעות בדידות לכללי הבחירה עם קבלה ונסיגה

נבנתה שיטת חיפוש שבה מודל MICA קודם מציע ערכים חדשים רק לבוררי כללי הניקוד
`sc_nb`, `sc_ch` ו־`sc_co`. כל הצעה נבדקת לבדה על עותק של מודל בקרה קצר,
באמצעות מנוע MICA השלם והמדויק. אם השיפור אינו מגיע לפחות ל־0.002
bits/target על 64 רשומות אימות נפרדות, השינוי נזרק והמודל המקורי נשמר.
הרשומות האלה אינן חלק ממערך האימות הנקי הקבוע.

נבדקו 46 קבוצות של שדה ושלב ועוד 16 קבוצות שבהן שלושת השדות מוחלפים יחד
בשלב אחד. ההצעה הטובה בקבוצה הראשונה שיפרה את נתוני הבחירה מ־2.647163
ל־2.645303; הטובה בקבוצה השנייה הגיעה ל־2.645215. שני השיפורים קטנים
מסף הקבלה, ולכן **לא התקבל אף שינוי**. קובצי המועמדים שנשמרו אחרי הנסיגה
זהים בבייטים למודל הבקרה.

לשם בדיקה בלתי תלויה ייצאתי את שתי ההצעות הטובות כמודלי בדיקה בלבד:

| 128 רשומות everyday נקיות | bits/target | פער לעומת הבקרה |
|---|---:|---:|
| בקרה קצרה | 2.558919 | — |
| הצעת שדה ושלב שנדחתה | 2.557985 | ‎−0.000935 |
| הצעת שלב משולבת שנדחתה | 2.562043 | ‎+0.003124 |
| המודל הטוב מאימון מלא קודם | **1.806367** | ‎−0.752552 |

רווח הסמך המזווג של ההצעה הראשונה הוא ‎[−0.002234, +0.000397], ולכן אין
עדות לשיפור יציב. גם כאן לא השתפרו המשפטים: ההצעה מפיקה
`The dog to come to the station with a cake with the water.` וכן
`The children man empty range the progreen to a couple of banananade...`.
שלוש מארבע ההשלמות הראשונות זהות לאלו של מודל הבקרה. השיטה שומרת על
ארכיטקטורת MICA ומונעת קידום של כללים שלא הוכיחו שיפור, אך הניסוי הזה לא
קירב את המודל ליעד של 1 bit/target או למשפטים הגיוניים חדשים.
ההשוואה למודל הטוב מאימון מלא נעשתה על אותן 128 רשומות; הפער של הצעת השדה
ממנו הוא ‎+0.751617 bits/target, רווח סמך מזווג
‎[+0.687082, +0.818060].

דוחות הניסויים, הנתונים והדוגמאות הגולמיות נמצאים תחת
`r1/runs/ablation/mica_reversible_state_pilot_20260927`,
`r1/runs/ablation/mica_reversible_full_20260927`,
`r1/runs/ablation/mica_reversible_rule_search_20260927`,
`r1/runs/ablation/mica_reversible2_pilot_20260927`,
`r1/runs/ablation/mica_probe_state_pilot_20260927`,
`r1/runs/ablation/mica_template_rule_pilot_20260927`,
`r1/runs/ablation/mica_informed_rule_pilot_20260927`,
`r1/runs/ablation/mica_template_phase_search_20260927` ו־
`r1/runs/ablation/mica_template_full_20260927`,
`r1/runs/ablation/mica_prose_pilot_20260927`,
`r1/runs/ablation/mica_exact_selector_acceptance_20260927` ו־
`r1/runs/ablation/mica_exact_selector_joint_20260927`.

## המשך בדיקת כללים וזיכרון בתוך MICA

חיפוש מדויק על המודל הטוב המלא הציע 96 שינויים בודדים לערכי כללים פעילים.
הטוב שבהם שיפר את קבוצת הקבלה הנפרדת רק מ־1.850644 ל־1.850040
bits/target, פחות מסף הקבלה 0.002. כל ההצעות נדחו; קובץ המועמד הסופי זהה
למודל הבקרה. [דוח החיפוש](../r1/runs/ablation/mica_full_active_rule_search_20260927/REPORT.md).

הכפלת מספר התאים שמתעדכנים בכל תו מאחד לשניים נתנה בניסוי קצר 2.555146
לעומת 2.549958 bits/target באימות נקי של 128 רשומות. גם בשני התאים
הקשר הרחוק לא השפיע על הניבוי לאחר סיומת משותפת של שמונה בתים.
[דוח חלון העדכון](../r1/runs/ablation/mica_window_context_pilot_20260927/REPORT.md).

נוסה זיכרון של שתי מילים באמצעות כללי המספרים השלמים של MICA עצמה, בלי
מנוע יצירה אחר. בדיקה מדויקת הראתה שמילים שונות עדיין משנות את התפלגות התו
הבא אחרי שמונה בתים זהים. למרות שהזיכרון עובד, סבב אימון מלא נתן **1.852494**
לעומת **1.826877** למודל ללא הזיכרון באותו תקציב, פער ‎+0.025617 עם רווח
סמך מזווג ‎[+0.018709,+0.032688]. זה גם מעל השיא 1.787362. לדוגמה,
`The doctor ` הושלם ל־`The doctor and a wooden bench with a large body of the
company of the company...`; משפטים חדשים והגיוניים טרם התקבלו.
[דוח הזיכרון](../r1/runs/ablation/mica_lexical_full_20260927/REPORT.md).

הפרדת אותיות לערוצי ניתוב קיימים, תוך שמירה על אותו מנוע, נתנה בפיילוט
2.554218 לעומת 2.551100 bits/target לגרסת הזיכרון הקודמת; רווח הסמך של
הפער כולל אפס. דוגמת פלט גולמי: `The dog large cause are train the street.`
[דוח ניתוב האותיות](../r1/runs/ablation/mica_letter_route_pilot_20260927/REPORT.md).

ניסוי המשך נפרד לתצורת MICA בעלת טווח בדיקה של 64 תווים החל בתיקייה
`r1/runs/overnight/mica_lag64_continuation_20260927`. סבב מלא ראשון שלה
השיג 1.797631 לעומת 1.817006 לתצורת 32 תווים תואמת. האימון ממשיך מנקודת
השמירה לסבב 40, עם הערכה מדויקת של המודל המיוצא ודוגמאות יצירה גולמיות
בסיום או מדי שעתיים, לפי המוקדם. אין קידום בלי תוצאה נקייה ואיכות משפטים.

### תוצאת סיום: המשך תצורת 64 התווים עד סבב 40

צילום שמור של המודל נמדד במנוע המספרים השלמים המדויק על אותן 1,000 רשומות
everyday נקיות של השיא הקודם: **1.765622 bits/target**, לעומת **1.787362**
לשיא הקודם ו־**1.797631** לתצורת 64 התווים אחרי סבב אחד. הפער המזווג לעומת
השיא הקודם הוא ‎−0.021740, רווח סמך bootstrap של 95%
‎[−0.029035, −0.014362]. זה שיא מספרי חדש *על מערך האימות הזה*, לא הגעה
ליעד 1 bit/target ולא אישור לאתר. הפלט הגולמי עדיין פגום:
`The doctor and a train the street.`;
`The dog is a computer and a train tracks of the street.`
הריצה הסתיימה בהצלחה בסבב 40. האימות הפנימי הטוב ביותר היה 1.7263 בסבב
20; סבבים 21–40 לא שיפרו את המודל השמור. דגימת הסיום של המנוע המדויק זהה
ב־SHA-256 לדגימת סבב 26 (`619147554e69`), ולכן תוצאת האימות הנקי נשארה
**1.765622**. לעומת סבב 1 של תצורת 64 התווים השיפור הוא ‎−0.032009,
רווח סמך מזווג ‎[−0.038665, −0.025156]. אין הצדקה להאריך מיד את אותו
אימון ללא שינוי שיטת בחירת הכללים: המדד הפנימי הפסיק להתקדם.

היצירה הישירה עדיין אינה עומדת ביעד המשפטים: 10 דוגמאות הפיתוח כוללות
חזרות של `the street signs on the street` ושיבושי תחביר. לא בוצע קידום
לאתר. הנתונים והפלט הגולמי של דגימת הסיום נשמרו תחת
`r1/runs/overnight/mica_lag64_continuation_20260927/samples/on_exit`.

### פיילוט בחירת היסטים מרוחקים לכללים (27 בספטמבר, 15:30–15:45 UTC)

נוסף דגל ניסוי `--rule-lag-coverage` לבחירה של היסטים שונים עבור תנאי
הכללים, במקום בחירות אקראיות שעלולות לחזור על אותו היסט. ההסקה נשארת במנוע
האוטומט התאי המקורי, עם כללים שלמים נלמדים. שני ענפי הפיילוט השתמשו באותם
40,000 קטעי אימון, seed, גאומטריה, סבב אחד, 1,500 צעדי התאמה וטווח
`rule-max-back=6`; ההבדל היחיד המתוכנן היה כיסוי ההיסטים. ההערכה נעשתה
במנוע המספרים השלמים המיוצא על כל `r1/data/eval_clean/val1000.jsonl`:
1,000 רשומות ו־44,795 מטרות (בתים ותו סיום, החל מ־BOS).

| ענף | קובץ מודל ו־SHA-256 | bits/target |
|---|---|---:|
| בחירה אקראית ארוכה | `r1/runs/codex_rule_lag_20260927/random_long/best.mica` · `053620041698ff7aa2b891aa6ae71789d153eedc7b13a600672e231660e921cd` | **2.158403** |
| כיסוי היסטים ארוכים | `r1/runs/codex_rule_lag_20260927/covered_long/best.mica` · `597832587100908476ac4e1d4ec6b43ca93a7bd6c8d22389eecdf219b5cc94ea` | 2.178223 |

הכיסוי החמיר ב־**0.019819 bits/target**, עם רווח סמך מזווג של 95%
‎[+0.007949, +0.031210]. לכן דגל הכיסוי אינו מועמד לקידום. שני המודלים
אומנו רק סבב אחד; אין להשוות את ערכיהם כמדד לתועלת של אימון ממושך מול השיא
הקודם, 1.765622 בסבב 40. גם איכות המשפטים בפיילוט אינה מספקת. שתי דוגמאות
גולמיות של ענף הכיסוי: `The dog in the side of a street signs on the side of a street signs on the side of a street signs on the sid` וכן
`Our flight the proton to do that.`. דוגמת ענף הבקרה:
`A man in a skateboard on a skateboard.`

בדיקות ההתאמה של הקוד עברו: `111 passed, 11 subtests passed`. הנתונים,
פרוטוקול ההשוואה וכל עשר היצירות הגולמיות לכל ענף שמורים תחת
`r1/runs/codex_rule_lag_20260927`. הועברה לקלוד בקשה לביקורת עצמאית על
הבחירה, בטיחות החידוש והביצועים. השלב הבא יהיה בחירה מונחית תוצאה של
פרדיקט וכלל יחד עם התאמה מחדש של הערך השלם, וקבלה רק לאחר אימות נקי מדויק.

### בחירה משותפת מוגבלת של תנאי כלל והערך השלם שלו

על המודל המוביל בעל SHA-256 `619147554e6927192612460152d66041f1e10ed6043688b8f7dfdcc1cfc6d9a1`
נבדקו 160 הצעות שמשנות יחד תנאי היסט של כלל פעיל ואת אחד מערכי ה־VSET
השלמים שהוא כותב. הבחירה וההתאמה נעשו על 32 רשומות *אימון* בלבד. ההצעה
המובילה שיפרה את 32 רשומות האימון מ־2.135317 ל־2.134719 bits/target,
אך על 64 רשומות פיתוח נפרדות שתי הגרסאות קיבלו בדיוק **1.878879**.
לכן ההצעה נדחתה והקובץ המיוצא זהה בבייטים למודל המקורי. תוצאת האימות הנקי
שלו נשארה **1.765622** על 1,000 רשומות ו־44,795 מטרות במנוע השלם המדויק;
לא הוכרז שיא חדש. שתי דוגמאות גולמיות של המודל שלא השתנה:
`The doctor and a train the street.` ו־
`The dog is a computer and a train tracks of the street.`
הפרטים נמצאים ב־`r1/runs/codex_joint_integer_rules_20260927_v2/REPORT.md`.

בביקורת ראשונית בלתי תלויה קלוד שיחזר את תוצאות פיילוט ההיסטים ומצא שגם
הרחבה גורפת של טווח הכללים לבדה מחמירה את התוצאה מול נקודת המוצא התואמת:
2.158403 לעומת 1.797631 בסבב אחד על אותו val1000. ממתינים לדוח הביקורת
המלא שלו. המסקנה המעשית היא לחפש התאמה של *קבוצת* כללים והקריאה שלהם,
ולא להאריך את טווח כל התנאים או לשנות כלל בודד ללא התאמה משותפת.
# Two-page selector rewrite, 17:49 UTC

The first coordinated pilot rewrote the two space-route rule pages (15 and 47)
with training-ranked three-back selectors and ran matched VSET/readout refits.
It was rejected: exact integer clean val1000 was **1.821897** for the proposal
(SHA `2acab39eff0be3f9b5cf5d5d34a66f1bfbbe2fc8650c3617fa7bcee6ce0d6fed`)
versus **1.765622** for the unchanged lag64 control (SHA `619147554e6927192612460152d66041f1e10ed6043688b8f7dfdcc1cfc6d9a1`),
1,000 records / 44,795 targets, bytes + EOS from BOS. The paired difference
was +0.056275 bits/target, 95% CI [+0.052180,+0.060404]. The disjoint dev128
was worse by +0.059817 [0.048117,0.070709]. The fit routine early-stopped
at the starting weights in both arms because its default learning rate raised
its own held-out loss. This outcome rules out the broad two-page rewrite as
tested, but does not test effective co-adaptation. Details and unedited raw
generations: `r1/runs/codex_page_block_refit_20260927/REPORT.md`.
# Small rule block plus low-rate refit, 18:02 UTC

A corrected matched pilot changed only 16 zero-win candidates in each of two
space-route pages and used `lr_scale=0.01` so the VSET/readout refit actually
optimized. The hard-selector proposal did **not** clear its independent dev128
acceptance gate: -0.000022 bits/target versus matched control, 95% paired CI
[-0.001273,+0.001285]. On exact integer clean val1000 the difference was
-0.000027, CI [-0.000583,+0.000520]. Candidate SHA
`c66c6beafecfbd33d591073c4791a9cc50e14682cefd00e6806074238877ae26`
was rejected.

The unchanged-selector control revealed a real improvement from refitting the
existing native MICA integer VSET values and readout at the lower rate. Its
SHA is `d6cedad89ba210efe4ee50c2b87d6989f9fff241b73c75718b401d14b6e127cc`
and exact clean everyday val1000 loss is **1.742303 bits/target** on 44,795
targets, versus 1.765622 for lag64; paired improvement -0.023319, 95% CI
[-0.026557,-0.020233]. This is a clean-loss best, not yet a sentence-quality
candidate: unedited generations still repeat phrases and produce incoherent
sentences. `r1/runs/codex_page_block_refit_20260927_v2/REPORT.md` records
protocol, model files, all raw generations, and conformance results. Next,
measure word prediction and sentence quality independently; pursue a learned
integer-rule state mechanism or a word-start objective based on those results.
# Word-start weighted native refits, 18:56 UTC

Matched refits of the accepted MICA checkpoint
`d6cedad89ba210efe4ee50c2b87d6989f9fff241b73c75718b401d14b6e127cc`
tested weights 1, 2 and 4 for the first byte after BOS or a space. On a fresh
disjoint dev256, weight 4 reduced exact integer word-start log loss by
0.010985 bits/start against the matched weight-1 arm, paired 95% CI
[0.006653,0.015225] for the improvement, but missed the predeclared 0.02
gain gate and reduced first-byte top-1 (24.88% to 24.59%). On clean val1000
it reached 1.740472 all-target bits/target and 3.764027 word-start bits on
8,445 starts, but whole-sentence greedy samples remained incoherent. The
weight-4 model SHA is
`82a4b94a9912d3023821961884dccd3428aaeb08661a652476759a88eec0c094`;
it was **not promoted** because dev acceptance failed. The last accepted
clean-loss best remains 1.742303. See
`r1/runs/codex_word_start_refit_20260927/REPORT.md` for protocol, all arms,
confidence intervals, and unedited generations. Next prioritize learned
integer-rule long-context state or a stronger rule-selection signal over
further first-byte weighting.

# Active-rule context split, 19:51 UTC

A rule actually used on the space-route page (1,682 wins in 2,000 training
records) was cloned into a dormant slot, then given a three-byte-back
condition that separated its parent wins with 95.9% purity. Both arms got
matched low-rate native VSET/readout refits. On **fresh disjoint dev256** the
split worsened exact integer loss by +0.000374 bits/target, paired 95% CI
[-0.000113,+0.000871], missing the predeclared -0.002 gate. On clean
everyday val1000 (44,795 byte-plus-EOS targets from BOS) the candidate
`r1/runs/codex_bucket_split_20260927/split/best.mica` SHA
`a2834d20c27dacf0d735731df238d1b1bf17370edaad2954c498c5c8b58e733f`
scored 1.740609 against matched control
`r1/runs/codex_bucket_split_20260927/control/best.mica` SHA
`a19f858edbf004e881743b183cba02de7b2584a3ced45a96f1b0ade2d9f39bb2`
at 1.740713; paired difference -0.000104 [-0.000340,+0.000111]. The CI
crosses zero, so the split is **rejected**. Fixed raw generations remained
identical and incoherent. The accepted best remains SHA `d6cedad...`
at 1.742303 pending stronger dev evidence for any later refit. Protocol,
exact files and two unedited English examples: 
`r1/runs/codex_bucket_split_20260927/REPORT.md`.

Claude's independent audit (`r1/runs/claude_audit_20260927/README.md`)
confirmed the clean scores and found that the earlier all-long rule-lag
pilot should be compared with its own one-round short-lag source at 1.797631:
all-long at 2.158403 was substantially worse, so long offsets should only
be trialed in selected phases while preserving phase 0/8 local context.
It also found that length-sorting provenance batches preserves their rows
while reducing padded work. Its proposed sort and independent coverage RNG
were applied to the fitter and all 111 tests plus 11 subtests passed.

# Selective long-context rule phases, 20:17 UTC

Following Claude's matched-control audit, a new one-round native MICA pilot
extended only phases 7, 11, 14 and 15 to the -8-byte scoring offset, while
keeping the earlier short phases. The unchanged-schedule arm reproduced its
one-round source checkpoint byte for byte. On fresh disjoint dev512, the
selective arm was **worse** by +0.018990 bits/target, paired 95% CI
[+0.010363,+0.027596]. On exact exported integer clean everyday val1000,
1,000 records / 44,795 byte-plus-EOS targets from BOS, its checkpoint
`r1/runs/codex_selective_long_phases_20260927/selective/best.mica` SHA
`44750015ef142385e6a34850a37f106814e909395d4f6d510b68114189e1a78b`
scored **1.825820** against matched control
`r1/runs/codex_selective_long_phases_20260927/control/best.mica` SHA
`24149019cc0a8cf16a83ce0444f60739691d50c813f9fa196219bb3425905679`
at **1.797631**; paired +0.028189 [+0.021559,+0.034769]. The proposal
was rejected, and raw greedy generations remained incoherent. Full
protocol, intervals and two unedited English samples are in
`r1/runs/codex_selective_long_phases_20260927/REPORT.md`.

Claude's fitter audit patch has now been applied: provenance batching sorts
records by length and restores row order, and lag coverage uses an independent
random stream while other rule fields remain matched. The local conformance
suite passed. The first matched one-round fits logged 31 s for provenance
on the GPU. Resume safety was also tightened: new checkpoints store the
initial rule settings, compatible older checkpoints check prior metadata,
and incompatible or unreadable state exits without overwriting a best model.
The complete suite now passes 113 tests and 11 subtests.

# Independent word and sentence assessment, 20:11 UTC

Claude evaluated the accepted clean-loss best `d6cedad8...` independently
on 2,000 next-word positions, 1,163 word completions after two letters,
and 100 blinded development prompts (`r1/runs/claude_audit_20260927/README.md`
§9). Next-word top-1 rose from 16.0% with lag64 to **16.7%**; completion
top-1 rose from 35.9% to **37.5%**. The position-level paired hits were
not retained, so these small differences are not statistically established.
The accepted MICA now roughly matches a 4.8 MB byte 6-gram on these two
word metrics, despite still having higher byte loss.

Blinded suggestion quality is still weak: **6% fully useful** and **18%
at least partly useful**, compared with lag64's 3% and 16% on the same
100 prompts; differences at this sample size are within rating noise.
Fifteen of 100 suggestions repeated ` back to the`, seven repeated
` a bathroom with`, and six repeated ` to do that.` across unrelated
contexts. Thus the clean-loss gain did not establish meaningful sentence
improvement. Further work must improve context-dependent rule decisions
and verify novelty and coherence separately from compression.

Claude's follow-up review identified two compatibility holes in the resume
safety change. Older `run_info.json` files omit options added later; their
historical defaults are now supplied only for that legacy comparison.
Repeated `run_train.py` sweep names now select a new output folder, retaining
earlier checkpoints. Conformance after those repairs: **115 passed, 11
subtests passed**.

# Information-trained phase-7 selector biases, 20:55 UTC

A bounded native hard-selector pilot changed 44 integer score biases across
23 phase-7 pages, selected by smoothed next-byte conditional entropy on
3,000 training records, followed by matched low-rate VSET/readout refits.
Claude's read-only audit found that the separate training screen was
measured but not used as a per-page gate, small splits were admitted, and
the proxy ignored the current model's other scoring terms and both read
lags. On fresh disjoint dev1000, exact exported integer MICA loss was
1.738865053 for the proposal versus 1.733729245 for its control:
**+0.005135808** bits/target, paired 95% CI
[+0.003541076,+0.006721942]. Word-start loss was also worse by
+0.010066546 bits/start. The predeclared development gate failed.

On clean everyday val1000 (1,000 records, 44,795 bytes-plus-EOS targets
from BOS), the proposal
`r1/runs/codex_phase_bias_info_20260927/proposal/best.mica` SHA
`2cec36eb9a34be4087351af4858717898bf8d69381046f543607787fd9ea6d06`
scored 1.745001339 versus matched control
`r1/runs/codex_phase_bias_info_20260927/control/best.mica` SHA
`a19f858edbf004e881743b183cba02de7b2584a3ced45a96f1b0ade2d9f39bb2`
at 1.740712749, paired **+0.004288589**
[+0.002554039,+0.006021757]. The selector proposal was rejected and
the accepted best remains d6cedad8 at 1.742303318. Raw generations
remain incoherent; two unedited examples and the full protocol are in
`r1/runs/codex_phase_bias_info_20260927/REPORT.md`.
