package week03

import breeze.linalg._
import helper.getMatrixFromFile
import scala.math.{E, log, pow}


trait LogisticRegression {
  def sigmoid(G: DenseMatrix[Double]): DenseMatrix[Double] = {
    def innerSigmoid(input: Double): Double = {
      1.0 /(1.0 + pow(E, -1.0 * input))
    }
      G.map(innerSigmoid)
  }

  // return cost and gradient as (cost, grad)
  def computeCost(X: DenseMatrix[Double],
                  Y: DenseMatrix[Double],
                  THETA: DenseMatrix[Double]): (Double, DenseMatrix[Double]) = {
    val setSize: Int = X.rows
    val sigmoidVec: DenseMatrix[Double] = sigmoid(X * THETA)
    val ERR: DenseMatrix[Double] = sigmoidVec - Y
    val dJ: DenseMatrix[Double] = (ERR.t * X) .* (1.0 / setSize)
    val cost: Double = sum(
      Y.t * sigmoidVec.map(log) +
        Y.map(x => - 1 * x + 1).t * sigmoidVec.map(x => log(- 1 * x + 1))
      ) / (setSize * (-1.0))
    (cost, dJ)
  }


  def predict(X: DenseMatrix[Double], THETA: DenseMatrix[Double]): Double = {
    val extendedX = DenseMatrix.horzcat(
      DenseMatrix.ones[Double](1, 1),
      X
    )
    val inp: DenseMatrix[Double] = DenseMatrix(sum(extendedX * THETA))
    sigmoid(inp)(0, 0)
  }

  val MAT: DenseMatrix[Double] = getMatrixFromFile("w03ex2data1.txt")
  val X: DenseMatrix[Double] = DenseMatrix.horzcat(
    DenseMatrix.ones[Double](MAT.rows, 1),
    MAT(0 to -1, 0 to 1)
  )
  val Y: DenseMatrix[Double] = MAT(0 to -1, 2 to 2)
  var THETA: DenseMatrix[Double] = DenseMatrix.zeros[Double](rows = X.cols, 1)

  val iterNum: Int = 3000000
  val alpha: Double = 0.4
  val setSize: Int = X.rows
  val J_history: DenseVector[Double] = DenseVector.zeros[Double](iterNum)
  var cost: Double = 0.0
  var dJ: DenseMatrix[Double] = DenseMatrix.zeros[Double](rows = X.cols, 1)
  var resCostFunc: (Double, DenseMatrix[Double]) = computeCost(X, Y, THETA)
  for(i <- 0 until iterNum){
    resCostFunc = computeCost(X, Y, THETA)
    cost = resCostFunc._1
    dJ = resCostFunc._2
    J_history(0) = cost
    print(cost)
    println
    THETA = THETA - dJ.t .* (alpha / setSize)
  }
  println("THETA: ")
  println(THETA)
  println("Predict when score1 = 45 and score2 = 85")
  println(
    predict(DenseMatrix(Array(45.0, 85.0)), THETA)
  )
}
