#!/bin/bash
# Job Name and Files
{{ site.sched.comment }} {{ site.sched.flag.name }} jobname

#Output and error:
{{ site.sched.comment }} {{ site.sched.flag.output }} ./%x.%j.out
{{ site.sched.comment }} {{ site.sched.flag.error }} ./%x.%j.err
#Initial working directory:
{{ site.sched.comment }} {{ site.sched.flag.directory }} ./

# Wall clock limit:
{{ site.sched.comment }} {{ site.sched.flag.time }}=00:15:00

#Setup of execution environment
{{ site.sched.comment }} {{ site.sched.flag.export }}
{{ site.sched.comment }} {{ site.sched.flag.env }}
{{ site.sched.comment }} {{ site.sched.flag.account }}="projectID"
{{ site.sched.comment }} {{ site.sched.flag.queue }}="Check Local System"

#Number of nodes and MPI tasks per node:
{{ site.sched.comment }} {{ site.sched.flag.node }}=1
{{ site.sched.comment }} {{ site.sched.flag.tasks_node }}=10

#Add 'Module load' statements here - check your local system for specific modules/names
{{ site.hemelb.env }}

#Run HemeLB 
rm -rf results #delete old results file
{{ site.sched.interactive}} -n {{ site.sched.ntasks }} {{ site.hemelb.exec }} -in input.xml -out results

