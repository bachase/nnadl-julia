TODO:
2. Broadcasting slow?  Store in cell? lines 76, 87
3. Why doesn't the AST figure this out?
4. Temporaries? Use !, any basic BLAS
5. Switch to looping over examples?
Devectorize
NumericExtensions
http://julialang.org/blog/2013/09/fast-numeric/

Done:
1. Reinterpret float32 to float64 -> ~2 seconds savings
2. NumericExtensions -> LogisticsFun
3. Instead of #2, just use vectorize_1arg