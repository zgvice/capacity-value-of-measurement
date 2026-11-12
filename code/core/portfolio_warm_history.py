"""相同历史线性约束复用求解基，不改变可行域与核验要求。"""
from types import SimpleNamespace
import numpy as np
import highspy
from portfolio_history import PortfolioHistory


class WarmPortfolioHistory(PortfolioHistory):
    def create_solver(self,objective,A,b,lower,upper):
        A=A.tocsr();h=highspy.Highs()
        h.setOptionValue('output_flag',False);h.setOptionValue('threads',self.threads)
        h.setOptionValue('solver','simplex')  # 同一线性规划重复更改功率上界，用单纯形基复用求解。
        h.setOptionValue('primal_feasibility_tolerance',self.c.solver_tolerance)
        h.setOptionValue('dual_feasibility_tolerance',self.c.solver_tolerance)
        lp=highspy.HighsLp();lp.num_col_=len(lower);lp.num_row_=len(b)
        lp.col_cost_=np.asarray(objective);lp.col_lower_=lower;lp.col_upper_=upper
        lp.row_lower_=np.full(len(b),-highspy.kHighsInf);lp.row_upper_=b
        lp.a_matrix_.format_=highspy.MatrixFormat.kRowwise;lp.a_matrix_.start_=A.indptr.astype(np.int32)
        lp.a_matrix_.index_=A.indices.astype(np.int32);lp.a_matrix_.value_=A.data
        h.passModel(lp)
        return h

    def result(self,h):
        h.run();status=h.getModelStatus()
        if status==highspy.HighsModelStatus.kInfeasible:return SimpleNamespace(success=False,status=2)
        if status!=highspy.HighsModelStatus.kOptimal:raise RuntimeError(h.modelStatusToString(status))
        solution=h.getSolution()
        return SimpleNamespace(success=True,status=0,x=np.asarray(solution.col_value),
            ineqlin=SimpleNamespace(marginals=np.asarray(solution.row_dual)))

    def lp(self,objective,A,b,lower,upper):
        return self.result(self.create_solver(objective,A,b,lower,upper))

    def solve(self,H):
        if np.any(np.asarray(H)<self.c.verified_min_duration_hours):return SimpleNamespace(success=False)
        key=tuple(np.asarray(H,float))
        if key in self.checked:return SimpleNamespace(success=self.checked[key])
        self.check_count+=1;upper=self.upper.copy();upper[:self.M*self.n]=self.cap(H)
        if not hasattr(self,'solver'):
            self.solver=self.create_solver(np.zeros(len(upper)),self.A,self.b,self.lower,upper)
            self.columns=np.arange(len(upper),dtype=np.int32)
        else:self.solver.changeColsBounds(len(upper),self.columns,self.lower,upper)
        result=self.result(self.solver);self.checked[key]=result.success
        return result
