import utils.tune_og as tune_og
import utils.tune_bootsrap as tune_bootsrap
import utils.tune_augment as tune_augment
import utils.tune_generated as tune_generated
import datetime


begin = datetime.datetime.now()

#tune_og.main()
#tune_bootsrap.main()
#tune_augment.main()
tune_generated.main()
end = datetime.datetime.now()

print("Experiment started at: ", begin.time())
print("Experiment ended at: ", end.time())
print("Total time: ", str(end - begin))