EXNAME=runme

SRC_C = ${EXNAME}.c
SRC_O = ${SRC_C:%.c=%.o}
SRC_D = ${SRC_C:%.c=%.d}

CFLAGS+=${C_DEPFLAGS}

all :: ${EXNAME}

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

${EXNAME}: ${SRC_O}
	-${CLINKER} -o $@ $^ ${PETSC_LIB}

clean :: 
	${RM} ${EXNAME} ${SRC_O} ${SRC_D}

# Indicate that SRC_D is up to date. Prevents the include from having quadratic complexity. 
$(SRC_D) : ; 
-include $(SRC_D)

.DELETE_ON_ERROR:
