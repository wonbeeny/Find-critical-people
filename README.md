# Find-critical-people
Sequence한 이동 경로와 밀접하게 연관된 사람을 찾습니다.

# Search_day.py
(1) Search_day_make_csv
- 보고 싶은 기간(15일)을 골라줌

# Find_P_make_csv.py
(1) Class : Find_methods
- data, sentence, top_n selection

(2) TF_sim
- Finding TF similarity

(3) count_rate
- Finding count rate = 해당 location을 들른 총 수 / 이동한 location 총 수

(4) D2V_sim
- Finding similarity using Doc2Vec model

(5) combind_method_count
- Using three methods, calculate two or three methods as output
