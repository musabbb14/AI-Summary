# System Usability Scale (SUS) Questionnaire

Used for Objective 4's qualitative evaluation (a minimum of ten
participants). After each participant completes a summarization task,
ask them to rate these 10 standard SUS statements from
1 (Strongly disagree) to 5 (Strongly agree).

1. I think that I would like to use this system frequently.
2. I found the system unnecessarily complex.
3. I thought the system was easy to use.
4. I think that I would need support to be able to use this system.
5. I found the various functions in this system were well integrated.
6. I thought there was too much inconsistency in this system.
7. I would imagine that most people would learn to use this system very quickly.
8. I found the system very cumbersome to use.
9. I felt very confident using the system.
10. I needed to learn a lot of things before I could get going with this system.

## Scoring
- For odd-numbered questions (1, 3, 5, 7, 9): score = rating - 1
- For even-numbered questions (2, 4, 6, 8, 10): score = 5 - rating
- Add all 10 scores, multiply by 2.5 -> SUS score out of 100
- A score above 68 is considered above-average usability

Record each participant's raw ratings in `sus_results.csv` using the
column layout in `sus_results_template.csv`.
