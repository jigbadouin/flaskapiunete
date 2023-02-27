#consulta para traer los resultados por pregunta de modulo

modulos = """
SELECT
  moduloUNO.userID,
  moduloUNO.PreguntaUno, moduloUNO.PreguntaDos, moduloUNO.PreguntaTres, moduloUNO.PreguntaCuatro,
  moduloDOS.PreguntaUno, moduloDOS.PreguntaDos, moduloDOS.PreguntaTres,
  moduloTRES.PreguntaUno, moduloTRES.PreguntaDos, moduloTRES.PreguntaTres, moduloTRES.PreguntaCuatro,
  moduloCUATRO.PreguntaUno, moduloCUATRO.PreguntaDos, moduloCUATRO.PreguntaTres,
  moduloCINCO.PreguntaUno, moduloCINCO.PreguntaDos, moduloCINCO.PreguntaTres,
  moduloSEIS.PreguntaUno, moduloSEIS.PreguntaDos, moduloSEIS.PreguntaTres, moduloSEIS.PreguntaCuatro, moduloSEIS.PreguntaCinco
  FROM moduloUNO
  INNER JOIN moduloDOS ON moduloDOS.userID = moduloUNO.userID
  INNER JOIN moduloTRES ON moduloTRES.userID = moduloUNO.userID
  INNER JOIN moduloCUATRO ON moduloCUATRO.userID = moduloUNO.userID
  INNER JOIN moduloCINCO ON moduloCINCO.userID = moduloUNO.userID
  INNER JOIN moduloSEIS ON moduloSEIS.userID = moduloUNO.userID
"""

modulosHeaders = ['IDUSer',
  'mod_UNO_p1', 'mod_UNO_p2', 'mod_UNO_p3', 'mod_UNO_p4',
  'mod_DOS_p1', 'mod_DOS_p2', 'mod_DOS_p3',
  'mod_TRES_p1', 'mod_TRES_p2', 'mod_TRES_p3', 'mod_TRES_p4',
  'mod_CUATRO_p1', 'mod_CUATRO_p2', 'mod_CUATRO_p3',
  'mod_CINCO_p1', 'mod_CINCO_p2', 'mod_CINCO_p3',
  'mod_SEIS_p1', 'mod_SEIS_p2', 'mod_SEIS_p3', 'mod_SEIS_p4', 'mod_SEIS_p5']

#consulta para traer los resultados por modulo y los resultados por variable
resultadosModVar = """
  SELECT
  resultados.modUNO, resultados.modDOS, resultados.modTRES, resultados.modCUATRO, resultados.modCINCO, resultados.modSEIS,
  resultadosVar.variableA, resultadosVar.variableB, resultadosVar.variableC, resultadosVar.variableD, resultadosVar.variableE,
  resultadosVar.variableF, resultadosVar.variableG, resultadosVar.variableI, resultadosVar.variableJ
  FROM resultados
  INNER JOIN resultadosVar ON resultadosVar.userID = resultados.userID
"""
resultadosModVarHeaders = ['UNO', 'DOS', 'TRES', 'CUATRO', 'CINCO', 'SEIS',
  'A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'J']

resultadosModHeaders = ['UNO', 'DOS', 'TRES', 'CUATRO', 'CINCO', 'SEIS']

resultadosModulos = """
  SELECT
  resultados.modUNO, resultados.modDOS, resultados.modTRES, resultados.modCUATRO, resultados.modCINCO, resultados.modSEIS
  FROM resultados
"""