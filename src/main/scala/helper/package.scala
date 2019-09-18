import breeze.linalg.{DenseMatrix, DenseVector}

import scala.annotation.tailrec
import scala.io.Source

package object helper {
  def parseLine(line: String, columnTerminator: String = ","): DenseVector[Double] = {
    val lst: Array[Double] = line.replace(" ", "")
      .split(columnTerminator)
      .map(x => x.toDouble)
    DenseVector(lst)
  }

  def getMatrixFromFile(resourcePath: String): DenseMatrix[Double] = {
    val rows = {for (line <- Source.fromResource(resourcePath).getLines()) yield parseLine(line)}.toList
    val rowLength: Int = rows.head.length
    @tailrec
    def go(ar: List[DenseVector[Double]], mat: DenseMatrix[Double]): DenseMatrix[Double] ={
      ar match {
        case Nil => mat
        case x :: xs => go(xs, DenseMatrix.vertcat(mat, x.toDenseMatrix))
      }
    }
    go(rows, DenseMatrix.zeros[Double](1, rowLength))(1 to -1, 0 to -1)
  }
}
