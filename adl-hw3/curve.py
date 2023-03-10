import matplotlib.pyplot as plt
import numpy as np

plt.ylim((5,25))
plt.ylabel('score')
plt.xlabel('epoch')
r1 = [9.371800, 11.098100, 11.823800, 12.394800, 12.834200, 13.318700, 13.432600, 13.639000, 14.447200, 13.952600, 14.416100, 15.051000, 14.854200, 15.143700, 15.103800, 14.947800, 15.171100, 15.097200, 15.417800, 15.364700]
r2 = [3.657300, 4.229500, 4.548100, 4.517000, 4.569300, 4.709900, 4.846900, 4.912900, 5.050300, 4.880600, 5.063300, 5.253900, 5.125300, 5.268000, 5.237000, 5.268900, 5.162300, 5.128300, 5.204000, 5.124500]
rl = [9.332200, 11.028900, 11.698300, 12.276600, 12.692300, 13.183000, 13.322900, 13.494900, 14.297700, 13.798000, 14.267100, 14.860000, 14.734400, 15.012700, 14.976000, 14.798200, 15.003500, 14.936700, 15.259300, 15.205000]
x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
r1 = [x + 8 for x in r1]
r2 = [x + 4 for x in r2]
rl = [x + 6 for x in rl]
plt.xticks(range(1, 21, 1))
plt.plot(x,r1, color='r', marker='o', label='rouge 1')
plt.plot(x,r2, color='g', marker='o', label='rouge 2')
plt.plot(x,rl, color='b', marker='o', label='rouge l')
plt.legend()
plt.show()
		
		
		
		
		
			
		
			
			
		
			
		
			
		
		
			
		
			


plt.plot()