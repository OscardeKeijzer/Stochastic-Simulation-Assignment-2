# -*- coding: utf-8 -*-
import simpy
import numpy as np
import math
import matplotlib.pyplot as plt
from datetime import datetime

# Define Customer class
class Customer(object):
    def __init__(self, id, env):    # Create instance
        self.id = id
        self.a_time = float(np.random.exponential(1/lamda))    # Sample arrival time from exponential distribution with rate lamda
        self.s_time = float(np.random.exponential(1/mu))    # Sample service time from exponential distribution with rate mu
        list_ids[s,n,j] = self.id    # Store in array
        list_a_time[s,n,j] = self.a_time    # Store in array
        list_s_time[s,n,j] = self.s_time    # Store in array
    
    def customer(self, env, id, serviceCenter, arrival_time, service_duration):    # Customer simulation function
        # Simulate arrival
        yield env.timeout(arrival_time)    # Yield until time of arrival has passed
        self.size_queue = len(serviceCenter.queue)    # Calculate size of queue on arrival for function prints
        list_q_size[s,n,j] = self.size_queue    # Store in array   
        # Request service
        #print('Arrival of customer %s at t = %.6s into a queue of %.6s' % (id, env.now, self.size_queue))
        with serviceCenter.request() as req:    # Request service
            yield req    # Yield until service available            
            # Receive service
            self.w_time = float(env.now - self.a_time)    # Calculate waiting time
            #print('After waiting %s, customer %s started receiving service at t = %.6s' % (self.w_time,id, env.now))            
            yield env.timeout(service_duration)
            #print('Customer %s left the service center at t = %.6s' % (id, env.now))
            self.r_time = float(self.w_time + self.s_time)    # Calculate response time

# Get time at start of simulation
start_time = datetime.now()
print(f'Simulation started at {start_time}.\n')

# Initialize simulation parameters
num_servers = 4    # Number of servers
num_sim = 10000    # Number of simulations per server
num_customers = 100    # Total number of customers to visit
lamda_initial = 2    # Arrival rate (lambda spelling is taken)
mu = 10    # Service rate

# Initialize queue variable lists
list_ids = np.array(num_sim*[num_servers*[num_customers*[0]]]).astype(int)    # Initialize list of customer ids
list_a_time = np.array(num_sim*[num_servers*[num_customers*[0]]]).astype(float)    # Initialize list of customer arrival times
list_s_time = np.array(num_sim*[num_servers*[num_customers*[0]]]).astype(float)    # Initialize list of customer service times
list_w_time = np.array(num_sim*[num_servers*[num_customers*[0]]]).astype(float)    # Initialize list of customer waiting times
list_r_time = np.array(num_sim*[num_servers*[num_customers*[0]]]).astype(float)    # Initialize list of customer response times
list_q_size = np.array(num_sim*[num_servers*[num_customers*[0]]]).astype(float)    # Initialize list of queue sizes

# Initialize statistical lists
list_a_mean = np.array(num_sim*[num_servers*[0]]).astype(float)    # Initialize list of means of arrival times in each simulation
list_s_mean = np.array(num_sim*[num_servers*[0]]).astype(float)    # Initialize list of means of service times in each simulation
list_w_mean = np.array(num_sim*[num_servers*[0]]).astype(float)    # Initialize list of means of waiting times in each simulation
list_r_mean = np.array(num_sim*[num_servers*[0]]).astype(float)    # Initialize list of means of response times in each simulation
list_q_mean = np.array(num_sim*[num_servers*[0]]).astype(float)    # Initialize list of means of queue sizes in each simulation
            
# Set up SimPy simulation
for s in range(num_sim):    # Number of simulations for each number of servers
    print('=====\nRunning simulation %s...' % (s+1))
    for n in range(num_servers):
        if n == 0 or n == 1 or n == 3:    # Run simulations for 1, 2, and 4 servers
            lamda = lamda_initial*(n+1)    # Update lamda to keep rho constant       
            env = simpy.Environment()    # Assign environment variable for each simulation
            serviceCenter = simpy.Resource(env, capacity=n+1)    # Initiate server resource with capacity the number of servers
            objs = np.empty(num_customers, dtype=object)    # Initialize array of customer objects
            for i in range(num_customers):    # Generate and simulate n customers
                j = i    # Customer counter for storing values into lists
                customer_ID = j + 1    # Create customer id
                objs[i] = Customer(customer_ID, env)    # Create ith customer object with corresponding id
                env.process(objs[i].customer(env,'Customer id %d' % objs[i].id, serviceCenter, objs[i].a_time, objs[i].s_time))    # Simulate ith customer
            env.run()    # Run simulation
            for i in range(num_customers):
                list_w_time[s,n,i] = objs[i].w_time
                list_r_time[s,n,i] = objs[i].r_time
                list_q_size[s,n,i] = objs[i].size_queue
            if s == num_sim - 1 and n == num_servers - 1 and j == num_customers - 1:
                print('\nLast customer left the service center at %s simulation time.' % env.now)
            
# Store estimator values in mean array
for s in range(num_sim):
    for n in range(num_servers):
        list_a_mean[s,n] = np.mean(list_a_time[s,n])
        list_s_mean[s,n] = np.mean(list_s_time[s,n])
        list_w_mean[s,n] = np.mean(list_w_time[s,n])
        list_r_mean[s,n] = np.mean(list_r_time[s,n])
        list_q_mean[s,n] = np.mean(list_q_size[s,n])

# Calculate mean, median, and variance of estimators
# 1 server:
estim_a1 = np.mean(list_a_mean[:,0])
median_a1 = np.median(list_a_mean[:,0])
var_a1 = np.var(list_a_mean[:,0])
estim_s1 = np.mean(list_s_mean[:,0])
median_s1 = np.median(list_s_mean[:,0])
var_s1 = np.var(list_s_mean[:,0])
estim_w1 = np.mean(list_w_time[:,0])
median_w1 = np.median(list_w_mean[:,0])
var_w1 = np.var(list_w_mean[:,0])
estim_r1 = np.mean(list_r_mean[:,0])
median_r1 = np.median(list_r_mean[:,0])
var_r1 = np.var(list_r_mean[:,0])
estim_q1 = np.mean(list_q_mean[:,0])
median_q1 = np.median(list_q_mean[:,0])
var_q1 = np.var(list_q_mean[:,0])
# 2 servers:
estim_a2 = np.mean(list_a_mean[:,1])
median_a2 = np.median(list_a_mean[:,1])
var_a2 = np.var(list_a_mean[:,1])
estim_s2 = np.mean(list_s_mean[:,1])
median_s2 = np.median(list_s_mean[:,1])
var_s2 = np.var(list_s_mean[:,1])
estim_w2 = np.mean(list_w_time[:,1])
median_w2 = np.median(list_w_mean[:,1])
var_w2 = np.var(list_w_mean[:,1])
estim_r2 = np.mean(list_r_mean[:,1])
median_r2 = np.median(list_r_mean[:,1])
var_r2 = np.var(list_r_mean[:,1])
# 4 servers:
estim_a4 = np.mean(list_a_mean[:,3])
median_a4 = np.median(list_a_mean[:,3])
var_a4 = np.var(list_a_mean[:,3])
estim_s4 = np.mean(list_s_mean[:,3])
median_s4 = np.median(list_s_mean[:,3])
var_s4 = np.var(list_s_mean[:,3])
estim_w4 = np.mean(list_w_time[:,3])
median_w4 = np.median(list_w_mean[:,3])
var_w4 = np.var(list_w_mean[:,3])
estim_r4 = np.mean(list_r_mean[:,3])
median_r4 = np.median(list_r_mean[:,3])
var_r4 = np.var(list_r_mean[:,3])

print(2*0.025*math.sqrt(np.var(list_w_time[0,0])/math.sqrt(num_sim)))
print(2*0.025*math.sqrt(var_w1)/math.sqrt(num_sim))

# Calculate total simulation time
print(f'\nThe simulation took {datetime.now()-start_time} (HH:mm:ss) to complete.\n')

# Plot results
print('Plotting results...')
# 1 server:
fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(ncols=2, nrows=2, figsize=(16,9))
fig1.suptitle('Results for %s FIFO Simulations With %s Server and %s Customers Each and ρ = %s' % (num_sim, 1, num_customers, lamda_initial/mu), size='xx-large')
ax1.hist(list_a_mean[:,0], bins='scott')
ax1.set_title('Mean Arrival Times')
ax1.set_xlabel('Mean Arrival Time (unit time)')
ax1.set_ylabel('Frequency')
textstr_a = '\n'.join((
    r'$\overline{E[A]}=%.6f$' % (estim_a1, ),
    r'$S=%.6f$' % (math.sqrt(var_a1), ),
    r'$\mathrm{median}=%.6f$' % (median_a1, )))
ax1.text(0.7, 0.95, textstr_a, transform=ax1.transAxes, fontsize=14,
        verticalalignment='top')
ax2.hist(list_s_mean[:,0], bins='scott')
ax2.set_title('Mean Service Times')
ax2.set_xlabel('Mean Service Time (unit time)')
ax2.set_ylabel('Frequency')
textstr_s = '\n'.join((
    r'$\overline{E[B}=%.6f$' % (estim_s1, ),
    r'$S=%.6f$' % (math.sqrt(var_s1), ),
    r'$\mathrm{median}=%.6f$' % (median_s1, )))
ax2.text(0.7, 0.95, textstr_s, transform=ax2.transAxes, fontsize=14,
        verticalalignment='top')
ax3.hist(list_w_mean[:,0], bins='scott')
ax3.set_title('Mean Waiting Times')
ax3.set_xlabel('Mean Waiting Time (unit time)')
ax3.set_ylabel('Frequency')
textstr_w = '\n'.join((
    r'$\overline{E[W]}=%.6f$' % (estim_w1, ),
    r'$S=%.6f$' % (math.sqrt(var_w1), ),
    r'$\mathrm{median}=%.6f$' % (median_w1, )))
ax3.text(0.7, 0.95, textstr_w, transform=ax3.transAxes, fontsize=14,
        verticalalignment='top')
ax4.hist(list_r_mean[:,0], bins='scott')
ax4.set_title('Mean Response Times')
ax4.set_xlabel('Mean Response Time (unit time)')
ax4.set_ylabel('Frequency')
textstr_r = '\n'.join((
    r'$\overline{E[T]}=%.6f$' % (estim_r1, ),
    r'$S=%.6f$' % (math.sqrt(var_r1), ),
    r'$\mathrm{median}=%.6f$' % (median_r1, )))
ax4.text(0.7, 0.95, textstr_r, transform=ax4.transAxes, fontsize=14,
        verticalalignment='top')
'''
ax5.plot(np.linspace(0,num_sim,1000), list_q_mean[:,0])
ax5.set_title('Mean Queue Sizes')
ax5.set_xlabel('Simulation Number')
ax5.set_ylabel('Mean Queue Size')
textstr_q = '\n'.join((
    r'$\overline{E[Q]}=%.6f$' % (estim_q1, ),
    r'$S=%.6f$' % (math.sqrt(var_q1), ),
    r'$\mathrm{median}=%.6f$' % (median_q1, )))
ax5.text(0.7, 0.25, textstr_q, transform=ax5.transAxes, fontsize=14,
        verticalalignment='top')'''
plt.tight_layout(pad=1.08, h_pad=None, w_pad=None, rect=None)
plt.subplots_adjust(top=0.9)
plt.show()
# 2 servers:
fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(ncols=2, nrows=2, figsize=(16,9))
fig2.suptitle('Results for %s FIFO Simulations With %s Servers and %s Customers Each and ρ = %s' % (num_sim, 2, num_customers, lamda_initial/mu), size='xx-large')
ax1.hist(list_a_mean[:,1], bins='scott')
ax1.set_title('Mean Arrival Times')
ax1.set_xlabel('Mean Arrival Time (unit time)')
ax1.set_ylabel('Frequency')
textstr_a = '\n'.join((
    r'$\overline{E[A]}=%.6f$' % (estim_a2, ),
    r'$S=%.6f$' % (math.sqrt(var_a2), ),
    r'$\mathrm{median}=%.6f$' % (median_a2, )))
ax1.text(0.7, 0.95, textstr_a, transform=ax1.transAxes, fontsize=14,
        verticalalignment='top')
ax2.hist(list_s_mean[:,1], bins='scott')
ax2.set_title('Mean Service Times')
ax2.set_xlabel('Mean Service Time (unit time)')
ax2.set_ylabel('Frequency')
textstr_s = '\n'.join((
    r'$\overline{E[B]}=%.6f$' % (estim_s2, ),
    r'$S=%.6f$' % (math.sqrt(var_s2), ),
    r'$\mathrm{median}=%.6f$' % (median_s2, )))
ax2.text(0.7, 0.95, textstr_s, transform=ax2.transAxes, fontsize=14,
        verticalalignment='top')
ax3.hist(list_w_mean[:,1], bins='scott')
ax3.set_title('Mean Waiting Times')
ax3.set_xlabel('Mean Waiting Time (unit time)')
ax3.set_ylabel('Frequency')
textstr_w = '\n'.join((
    r'$\overline{E[W]}=%.6f$' % (estim_w2, ),
    r'$S=%.6f$' % (math.sqrt(var_w2), ),
    r'$\mathrm{median}=%.6f$' % (median_w2, )))
ax3.text(0.7, 0.95, textstr_w, transform=ax3.transAxes, fontsize=14,
        verticalalignment='top')
ax4.hist(list_r_mean[:,1], bins='scott')
ax4.set_title('Mean Response Times')
ax4.set_xlabel('Mean Response Time (unit time)')
ax4.set_ylabel('Frequency')
textstr_r = '\n'.join((
    r'$\overline{E[T]}=%.6f$' % (estim_r2, ),
    r'$S=%.6f$' % (math.sqrt(var_r2), ),
    r'$\mathrm{median}=%.6f$' % (median_r2, )))
ax4.text(0.7, 0.95, textstr_r, transform=ax4.transAxes, fontsize=14,
        verticalalignment='top')
plt.tight_layout(pad=1.08, h_pad=None, w_pad=None, rect=None)
plt.subplots_adjust(top=0.9)
plt.show()
# 4 servers:
fig3, ((ax1, ax2), (ax3, ax4)) = plt.subplots(ncols=2, nrows=2, figsize=(16,9))
fig3.suptitle('Results for %s FIFO Simulations With %s Servers and %s Customers Each and ρ = %s' % (num_sim, 4, num_customers, lamda_initial/mu), size='xx-large')
ax1.hist(list_a_mean[:,3], bins='scott')
ax1.set_title('Mean Arrival Times')
ax1.set_xlabel('Mean Arrival Time (unit time)')
ax1.set_ylabel('Frequency')
textstr_a = '\n'.join((
    r'$\overline{E[A]}=%.6f$' % (estim_a4, ),
    r'$S=%.6f$' % (math.sqrt(var_a4), ),
    r'$\mathrm{median}=%.6f$' % (median_a4, )))
ax1.text(0.7, 0.95, textstr_a, transform=ax1.transAxes, fontsize=14,
        verticalalignment='top')
ax2.hist(list_s_mean[:,3], bins='scott')
ax2.set_title('Mean Service Times')
ax2.set_xlabel('Mean Service Time (unit time)')
ax2.set_ylabel('Frequency')
textstr_s = '\n'.join((
    r'$\overline{E[B]}=%.6f$' % (estim_s4, ),
    r'$S=%.6f$' % (math.sqrt(var_s4), ),
    r'$\mathrm{median}=%.6f$' % (median_s4, )))
ax2.text(0.7, 0.95, textstr_s, transform=ax2.transAxes, fontsize=14,
        verticalalignment='top')
ax3.hist(list_w_mean[:,3], bins='scott')
ax3.set_title('Mean Waiting Times')
ax3.set_xlabel('Mean Waiting Time (unit time)')
ax3.set_ylabel('Frequency')
textstr_w = '\n'.join((
    r'$\overline{E[W]}=%.6f$' % (estim_w4, ),
    r'$S=%.6f$' % (math.sqrt(var_w4), ),
    r'$\mathrm{median}=%.6f$' % (median_w4, )))
ax3.text(0.7, 0.95, textstr_w, transform=ax3.transAxes, fontsize=14,
        verticalalignment='top')
ax4.hist(list_r_mean[:,3], bins='scott')
ax4.set_title('Mean Response Times')
ax4.set_xlabel('Mean Response Time (unit time)')
ax4.set_ylabel('Frequency')
textstr_r = '\n'.join((
    r'$\overline{E[T]}=%.6f$' % (estim_r4, ),
    r'$S=%.6f$' % (math.sqrt(var_r4), ),
    r'$\mathrm{median}=%.6f$' % (median_r4, )))
ax4.text(0.7, 0.95, textstr_r, transform=ax4.transAxes, fontsize=14,
        verticalalignment='top')
plt.tight_layout(pad=1.08, h_pad=None, w_pad=None, rect=None)
plt.subplots_adjust(top=0.9)
plt.show()