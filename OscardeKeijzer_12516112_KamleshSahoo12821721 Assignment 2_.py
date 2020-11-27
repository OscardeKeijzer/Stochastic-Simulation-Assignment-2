# -*- coding: utf-8 -*-
import simpy
import numpy as np
import matplotlib.pyplot as plt


# Initialize simulation parameters
n = 5    # Total number of customers
c = 0    # Customer id counter
lamda = 5    # Arrival rate (lambda spelling is taken)
mu = 10    # Service rate
list = []    # List of customers

# Define arrival and service functions
def a(t):    # a(t) function
    a = lamda*np.exp(-lamda*t)
    return a

def A(t):    # A(t) function
    A = 1 - np.exp(-lamda*t)
    return A

def b(t):    # b(t) function
    b = mu*np.exp(-mu*t)
    return b

def B(t):    # B(t) function
    B = 1 - np.exp(-mu*t)
    return B

# Customer class
class Customer(object):
    def __init__(self, id, env):    # Create instance
        self.id = id
        self.a_time = a(np.random.random(1)[0])    # Sample arrival time from exponential distribution with rate lamda
        self.s_time = b(np.random.random(1)[0])    # Sample service time from exponential distribution with rate mu

    def customer(self, env, id, serviceCenter, arrival_time, service_duration):    # Customer simulation function
        # Simulate arrival
        yield env.timeout(arrival_time)
        
        # Request service
        print('Arrival of %s at t = %s' % (id, env.now))
        with serviceCenter.request() as req:
            yield req
            
            # Receive service
            print('%s started receiving service at t = %s' % (id, env.now))
            yield env.timeout(service_duration)
            print('%s left the service center at t = %s' % (id, env.now))

# Initialize SimPy simulation
env = simpy.Environment()    # Initiate the SimPy environment
serviceCenter = simpy.Resource(env, capacity=1)    # Initiate server resource, capacity = number of servers

for i in range(n):    # Generate and simulate n customers
    c = c + 1    # Create customer id c (NOT directly related to arrival time)
    i = Customer(c, env)    # Create ith customer object with id c
    env.process(i.customer(env,'Customer id %d' % i.id, serviceCenter, i.a_time, i.s_time))
    list.append(i)

env.run()    # Run SimPy simulation

# Print simulation results
print('\n')

for i in list:
    print(('Customer id %d arrival time was t = %s and service time was %s') % (i.id, i.a_time, i.s_time))
    
print('\nMean customer arrival time was %s ' % ((sum(i.a_time for i in list)/n)))

'''
# Plot a(t) and A(t) functions
t = np.linspace(0,1)
plt.scatter(t,a(t), label='$\mathit{a(t)}$')
plt.scatter(t,A(t), label='$\mathit{A(t)}$')
plt.scatter(t,b(t), label='$\mathit{b(t)}$')
plt.scatter(t,B(t), label='$\mathit{B(t)}$')
plt.legend()
plt.xlabel('$\mathit{t}$')
'''