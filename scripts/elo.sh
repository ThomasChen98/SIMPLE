sudo docker-compose exec app mpirun -np 8 python3 elo.py -e tictactoe -r -g 200000 -a 1 272 5 -p 8 -ld data/SP_TTT_20M_s8/models -sn tictactoe_SP_s8
# sudo docker-compose exec app mpirun -np 3 python3 elo.py -e tictactoe -r -g 200000 -a 1 242 5 -p 3 -ld data/PP_TTT_20M_s3/models -sn tictactoe_PP_s3
# sudo docker-compose exec app mpirun -np 5 python3 elo.py -e tictactoe -r -g 200000 -a 1 149 5 -p 5 -ld data/PP_TTT_20M_s5/models -sn tictactoe_PP_s5
# sudo docker-compose exec app mpirun -np 3 python3 elo.py -e tictactoe -r -g 200000 -a 1 139 5 -p 3 -ld data/PP_TTT_20M_s5/models -sn tictactoe_FCP_s3
# sudo docker-compose exec app mpirun -np 1 python3 elo.py -e tictactoe -r -g 200000 -a 1 445 10 -p 1 -ld data/SP_TTT_20M_s8/models -sn tictactoe_SP_s1