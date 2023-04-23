sudo docker-compose exec app mpirun -np 25 python3 pvalue.py -e tictactoe -r -g 20 -ld data
# sudo docker-compose exec app python3 pvalue.py -e tictactoe -r -g 20 -ld data -l tictactoe_g20.npz