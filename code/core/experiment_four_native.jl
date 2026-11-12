using PRAS, Dates, DelimitedFiles, JSON3
# 路径由调用入口传入；输入来自experiment_four_physics.py/current 15分钟年度场景。
const DEST=ARGS[1]
const TCASE=105120 # 原完整8760小时，每小时十二个5分钟；原15分钟功率每三个点相同。
data=readdlm(joinpath(DEST,"pras_input.csv"),',',Int;skipstart=1)
rows=Any[]
for id in sort(unique(data[:,1]))
    z=data[data[:,1].==id,:]
    @assert size(z,1)==TCASE
    # 原生整数数值以kW输入；除1000恢复MW；每期5分钟，电量再除12。
    capacity=Matrix(transpose(z[:,4:5]));ld=z[:,3]
    generators=Generators{TCASE,5,Minute,MW}(["same_system_path","same_customer_path"],["fixed","fixed"],capacity,zeros(2,TCASE),ones(2,TCASE))
    times=ZonedDateTime(2027,6,1,tz"UTC"):Minute(5):ZonedDateTime(2027,6,1,tz"UTC")+Minute(5*(TCASE-1)) # 模型时序索引。
    system=SystemModel(generators,Storages{TCASE,5,Minute,MW,MWh}(),GeneratorStorages{TCASE,5,Minute,MW,MWh}(),times,ld)
    # 两次确定性记录用于工具采样接口；同一条输入没有增加独立年度样本。
    result,=assess(system,SequentialMonteCarlo(samples=2,seed=1,threaded=false),ShortfallSamples())
    observed=vec(result.shortfall[1,:,1]);gap=maximum(abs.(observed.-z[:,6]))
    @assert gap==0
    push!(rows,Dict("case"=>id,"steps"=>TCASE,"minutes"=>5,"hours"=>8760,"max_shortfall_difference_MW"=>gap/1000,"native_unserved_MWh"=>sum(observed)/12000))
    println("PRAS annual case=",id," gap_MW=",gap/1000);flush(stdout)
end
open(joinpath(DEST,"pras_results.json"),"w") do f
    JSON3.pretty(f,Dict("status"=>"completed","julia_version"=>string(VERSION),"cases"=>rows))
end
