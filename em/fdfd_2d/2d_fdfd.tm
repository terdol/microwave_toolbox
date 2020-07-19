<TeXmacs|1.99.5>

<style|<tuple|old-generic|math-brackets>>

<\body>
  <\equation*>
    <frac|\<beta\>|k<rsub|o>>E<rsub|x><around*|(|i,j|)>=H<rsub|y><around*|(|i,j|)>+<frac|j|k<rsub|o>dx<around*|(|i|)>><around*|[|E<rsub|z><around*|(|i+1,j|)>-E<rsub|z><around*|(|i,j|)>|]>
  </equation*>

  <\equation*>
    <frac|\<beta\>|k<rsub|o>>E<rsub|x><around*|(|i,j|)>=H<rsub|y><around*|(|i,j|)>+<frac|j|k<rsub|o>dx<around*|(|i|)>><around*|[|<frac|-j|k<rsub|o>ddx<around*|(|i|)>\<varepsilon\><rsub|zz><around*|(|i+1,j|)>><around*|[|H<rsub|y><around*|(|i+1,j|)>-H<rsub|y><around*|(|i,j|)>|]>+<frac|j|k<rsub|o>ddy<around*|(|j-1|)>\<varepsilon\><rsub|zz><around*|(|i+1,j|)>><around*|[|H<rsub|x><around*|(|i+1,j|)>-H<rsub|x><around*|(|i+1,j-1|)>|]>-<around*|(|<frac|-j|k<rsub|o>ddx<around*|(|i-1|)>\<varepsilon\><rsub|zz><around*|(|i,j|)>><around*|[|H<rsub|y><around*|(|i,j|)>-H<rsub|y><around*|(|i-1,j|)>|]>+<frac|j|k<rsub|o>ddy<around*|(|j-1|)>\<varepsilon\><rsub|zz><around*|(|i,j|)>><around*|[|H<rsub|x><around*|(|i,j|)>-H<rsub|x><around*|(|i,j-1|)>|]>|)>|]>
  </equation*>

  <\equation*>
    ddx<around*|(|i|)>=<frac|dx<around*|(|i|)>+dx<around*|(|i+1|)>|2><space|3em>ddy<around*|(|j|)>=<frac|dy<around*|(|j|)>+dy<around*|(|j+1|)>|2>
  </equation*>

  <\equation*>
    <frac|\<beta\>|k<rsub|o>>E<rsub|x><around*|(|i,j|)>=H<rsub|y><around*|(|i,j|)>+<frac|1|k<rsub|o><rsup|<rsup|2>>ddx<around*|(|i|)>dx<around*|(|i|)>\<varepsilon\><rsub|zz><around*|(|i+1,j|)>><around*|[|H<rsub|y><around*|(|i+1,j|)>-H<rsub|y><around*|(|i,j|)>|]>-<frac|1|k<rsub|o><rsup|<rsup|2>>dx<around*|(|i|)>ddy<around*|(|j-1|)>\<varepsilon\><rsub|zz><around*|(|i+1,j|)>><around*|[|H<rsub|x><around*|(|i+1,j|)>-H<rsub|x><around*|(|i+1,j-1|)>|]>-<around*|\<nobracket\>|<frac|1|k<rsub|o><rsup|<rsup|2>>ddx<around*|(|i-1|)>dx<around*|(|i|)>\<varepsilon\><rsub|zz><around*|(|i,j|)>><around*|[|H<rsub|y><around*|(|i,j|)>-H<rsub|y><around*|(|i-1,j|)>|]>+<frac|1|k<rsub|o><rsup|<rsup|2>>dx<around*|(|i|)>ddy<around*|(|j-1|)>\<varepsilon\><rsub|zz><around*|(|i,j|)>><around*|[|H<rsub|x><around*|(|i,j|)>-H<rsub|x><around*|(|i,j-1|)>|]>|)>
  </equation*>

  \;

  \;

  <with|math-display|true|<with|math-condensed|true|<math|<frac|\<beta\>|k<rsub|o>>E<rsub|y><around*|(|i,j|)>=-H<rsub|x><around*|(|i,j|)>+<frac|j|k<rsub|o>dy<around*|(|j|)>><around*|[|E<rsub|z><around*|(|i,j+1|)>-E<rsub|z><around*|(|i,j|)>|]>>>>

  <\equation*>
    <frac|\<beta\>|k<rsub|o>>E<rsub|y><around*|(|i,j|)>=-H<rsub|x><around*|(|i,j|)>+<frac|j|k<rsub|o>dy<around*|(|j|)>><around*|[|<frac|-j|k<rsub|o>ddx<around*|(|i-1|)>\<varepsilon\><rsub|zz><around*|(|i,j+1|)>><around*|[|H<rsub|y><around*|(|i,j+1|)>-H<rsub|y><around*|(|i-1,j+1|)>|]>+<frac|j|k<rsub|o>ddy<around*|(|j|)>\<varepsilon\><rsub|zz><around*|(|i,j|)>><around*|[|H<rsub|x><around*|(|i,j+1|)>-H<rsub|x><around*|(|i,j|)>|]>-<around*|(|<frac|-j|k<rsub|o>ddx<around*|(|i-1|)>\<varepsilon\><rsub|zz><around*|(|i,j|)>><around*|[|H<rsub|y><around*|(|i,j|)>-H<rsub|y><around*|(|i-1,j|)>|]>+<frac|j|k<rsub|o>ddy<around*|(|j-1|)>\<varepsilon\><rsub|zz><around*|(|i,j|)>><around*|[|H<rsub|x><around*|(|i,j|)>-H<rsub|x><around*|(|i,j-1|)>|]>|)>|]>
  </equation*>

  \;

  \;

  <math|<frac|\<beta\>|k<rsub|o>>E<rsub|y><around*|(|i,j|)>=-H<rsub|x><around*|(|i,j|)>+<frac|1|k<rsub|o><rsup|2>dy<around*|(|j|)>ddx<around*|(|i-1|)>\<varepsilon\><rsub|zz><around*|(|i,j+1|)>><around*|[|H<rsub|y><around*|(|i,j+1|)>-H<rsub|y><around*|(|i-1,j+1|)>|]>-<frac|1|k<rsub|o><rsup|<rsup|2>>dy<around*|(|j|)>ddy<around*|(|j|)>\<varepsilon\><rsub|zz><around*|(|i,j|)>><around*|[|H<rsub|x><around*|(|i,j+1|)>-H<rsub|x><around*|(|i,j|)>|]><around*|\<nobracket\>|-<frac|1|k<rsub|o><rsup|<rsup|2>>dy<around*|(|j|)>ddx<around*|(|i-1|)>\<varepsilon\><rsub|zz><around*|(|i,j|)>><around*|[|H<rsub|y><around*|(|i,j|)>-H<rsub|y><around*|(|i-1,j|)>|]>+<frac|1|k<rsub|o><rsup|<rsup|2>>dy<around*|(|j|)>ddy<around*|(|j-1|)>\<varepsilon\><rsub|zz><around*|(|i,j|)>><around*|[|H<rsub|x><around*|(|i,j|)>-H<rsub|x><around*|(|i,j-1|)>|]>|)>>

  \;

  Bu formüllerde ³u normalizasyon uygulanm\Y³:

  <\equation*>
    H<rprime|'>=H*<sqrt|\<eta\><rsub|0>><space|3em>E<rprime|'>=<frac|E|<sqrt|\<eta\><rsub|0>>>
  </equation*>

  \;

  \;

  \;

  \;

  \;
</body>

<\initial>
  <\collection>
    <associate|page-orientation|landscape>
    <associate|page-type|a2>
  </collection>
</initial>