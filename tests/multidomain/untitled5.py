# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 18:26:54 2024

@author: Nathan
"""
current_1 = 'MX1'
platoons_current_1 = 4
current_2 = 'MX2'
platoons_current_2 = 5
preload_1 = 'None'
preload_2 = ''

if current_1[-1] == '1':
    TP_1 = 10.0
if current_1[-1] == '2':
    TP_1 = 11.0
if current_1[-1] == '3':
    TP_1 = 13.5
if current_2[-1] == '1':
    TP_2 = 10.0
if current_2[-1] == '2':
    TP_2 = 11.0
if current_2[-1] == '3':
    TP_2 = 13.5

print("<@&790522194341789707>")
print()
print('# PLAN OF ACTION:')
if current_2 is not None:
    print(f'- **3* both {current_1} and {current_2}** then preload {preload_1}')
else:
    print(f'- **3* {current_1}** then preload {preload_1} and {preload_2}')

print('# PLATOONS:')
print('## OPEN TO ALL - Proritize after assigned units')
print(f'- **{current_1}**: We can do **{platoons_current_1}**')
TP_total = platoons_current_1*TP_1
print(f'That is {platoons_current_1} x {TP_1} = {TP_total} mil per planet')

if current_2 is not None:
    print(f'- **{current_2}**: We can do **{platoons_current_2}**')
    TP_total = platoons_current_2*TP_2
    print(f'That is {platoons_current_2} x {TP_2} = {TP_total} mil per planet')

print('## ORDERS ONLY')
print('Only place if you have orders from the HotBot. Otherwise ignore')
print(f'- {preload_1}')
if current_2 is None:
    print(f'- {preload_2}')

print('# FLEET CMs')
print(f'**{current_1}** : ')

# print# FLEET CMs:
# print- **LS1:** Profundity (full auto) / Home One with Outrider 
# print- **MX1**: Exec with Landos ship
# print- **DS1**: Tarkin or Thrawn with scythe  

# printFor other CMs, run /tb combat in <#941688013958819891>  . If it shows you a team, at the very least please put it on auto and leave it.
# printYou can also check the recommended squads at https://genskaar.github.io/tb_empire/html/main.html