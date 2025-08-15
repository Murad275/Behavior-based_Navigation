import time
import casadi as ci
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
import numpy as np
import scipy.linalg
import math

class filter_mpc(object):
    def __init__(self):
        self.RUN_ONCE = True
        self.tcomp_sum = 0
        self.tcomp_max = 0
        self.acados_solver = 0
        self.N = 0
        self.Tf = 0
        self.active = False
        self.U_calc = []
        

    def kine_model(self):

        model = ci.types.SimpleNamespace()
        model_name = 'kine_ode'

        x = ci.SX.sym('x')
        y = ci.SX.sym('y')
        phi = ci.SX.sym('phi')
        
               
        # controls
        v = ci.SX.sym('v')
        phidot = ci.SX.sym('phidot')
        
        # statesdot
        xdot  = ci.SX.sym('xdot')
        ydot  = ci.SX.sym('ydot')
        phid = ci.SX.sym('phid')
                
        obs_x = ci.SX.sym('obs_x')
        obs_y = ci.SX.sym('obs_y')
        xg = ci.SX.sym('xg')
        yg = ci.SX.sym('yg')
        x_1 = ci.SX.sym('x_1')
        y_1 = ci.SX.sym('y_1')
        x_2 = ci.SX.sym('x_2')
        y_2 = ci.SX.sym('y_2')

        states = ci.vertcat(x, y, phi)
        statesdot = ci.vertcat(xdot, ydot, phid)
        u = ci.vertcat(v, phidot)    
      
        dist = ci.sqrt((x - obs_x)**2 + (y - obs_y)**2)
        dist1 = ci.sqrt((x - x_1)**2 + (y - y_1)**2)
        dist2 = ci.sqrt((x - x_2)**2 + (y - y_2)**2)

        
        fn_dist = 1/(ci.exp(10*dist)) # 10 is a factor to be tuned based on the robots and lidars
        fn_dist1 = 1/(ci.exp(10*dist1)) 
        fn_dist2 = 1/(ci.exp(10*dist2))


        p = ci.vertcat(obs_x, obs_y, x_1, y_1, x_2, y_2)
        
        f_expl = ci.vertcat(v*ci.cos(phi),
                        v*ci.sin(phi),
                        phidot
                        )


        model = AcadosModel()
        model.f_expl_expr = f_expl
        model.x = states
        model.xdot = statesdot
        model.u = u
        model.p = p
        model.name = model_name
                
        model.cost_y_expr = ci.vertcat(u, fn_dist,fn_dist1, fn_dist2)
        model.cost_y_expr_e = ci.vertcat(fn_dist,fn_dist1, fn_dist2)     
        return model

    ##############################################    
    def acado_set(self, x0,i):
        # create ocp object to formulate the OCP
        ocp = AcadosOcp()
        self.Tf = 2
        self.N = 20
        self.times = []


        # set model
        model = self.kine_model()

        nu = model.u.size()[0]
        ny = nu + 3
        nye = 3
        npa = model.p.size()[0]


        ocp.model = model
        # set dimensions
        ocp.dims.N = self.N
        ocp.dims.np = npa

        # set cost module
        ocp.cost.cost_type = 'NONLINEAR_LS'
        ocp.cost.cost_type_e = 'NONLINEAR_LS'
          
        ocp.parameter_values = np.array([5, 5, 5, 5, 5, 5])
        
        w_dist = 4 # To be tuned based on the robots and lidars

        R1 = np.diag([10, 10]) # To be tuned based on the robots and lidars
        ocp.cost.W_0 = scipy.linalg.block_diag(R1, w_dist,w_dist,w_dist)

        R2 = np.diag([0.001, 0.001]) 
        ocp.cost.W = scipy.linalg.block_diag(R2, w_dist,w_dist,w_dist)

        ocp.cost.W_e = scipy.linalg.block_diag(w_dist,w_dist,w_dist)

        ocp.cost.yref  = np.zeros((ny, ))
        ocp.cost.yref_e  = np.zeros((nye, ))
        # set constraints
        vmax = 0.75
        phidotmax = 0.75
        # ocp.constraints.constr_type = 'BGH'
        
        ocp.constraints.idxbu = np.array([0, 1]) 
        ocp.constraints.lbu = np.array([0 , -phidotmax])
        ocp.constraints.ubu = np.array([vmax, phidotmax])
        
        ocp.constraints.idxbx = np.array([0, 1, 2])
        ocp.constraints.lbx = np.array([-4.65, -4.65, -math.pi]) # limits of environment, change based on environment
        ocp.constraints.ubx = np.array([4.65, 4.65, math.pi])



        ocp.constraints.x0 = x0
        
        ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM' 
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.nlp_solver_type = 'SQP' # SQP_RTI
        ocp.solver_options.sim_method_num_stages = 4
        ocp.solver_options.sim_method_num_steps = 3
        ocp.solver_options.qp_solver_cond_N = self.N

        

        ocp.solver_options.tf = self.Tf

        acados_ocp_solver = AcadosOcpSolver(ocp, json_file = 'acados_ocp.json'+str(i)+'real')
        
        return  acados_ocp_solver
    ############################################

    def run_mpc(self, u_L, x0, obs_x, obs_y, x_1, y_1, x_2,y_2, i):
        
        if self.RUN_ONCE:
            self.acados_solver = self.acado_set(x0,i)
            self.U_calc = np.zeros((self.N, 2))
            self.RUN_ONCE = False
           
        ydes = np.array(u_L)
        yrf = np.append(ydes,[0,0,0])
        self.acados_solver.set(0, "yref", yrf)
        self.acados_solver.set(0,"p", np.array([obs_x, obs_y, x_1, y_1, x_2,y_2]))
        self.acados_solver.set(0,"x", x0)
        self.acados_solver.set(0,"u", np.array(u_L))
        self.acados_solver.set(0, "lbx", x0)
        self.acados_solver.set(0, "ubx", x0)

        
        for j in range(1,self.N):
            self.acados_solver.set(j, "yref", np.zeros(5))
            self.acados_solver.set(j,"p", np.array([obs_x, obs_y, x_1, y_1, x_2,y_2 ]))
            # self.acados_solver.set(j,"x", np.zeros(3))
            self.acados_solver.set(j,"u", self.U_calc[j])            

                        
        yref_N = np.zeros(3)
        self.acados_solver.set(self.N, "yref", yref_N)
        self.acados_solver.set(self.N,"p", np.array([obs_x, obs_y,x_1, y_1, x_2,y_2]))
        self.acados_solver.set(self.N,"x", np.zeros(3))
        # solve ocp
        t = time.time()

        status = self.acados_solver.solve()
        self.times.append( time.time() - t )
        
        
        if status != 0:
            u0 = [0,0]
        else:
            u0 = self.acados_solver.get(0, "u")

        return u0
