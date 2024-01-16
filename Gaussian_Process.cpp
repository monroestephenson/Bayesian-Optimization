#include <Eigen/Dense>
using namespace Eigen;

class GaussianProcess {
public:
    // Define your kernel function as needed
    class KernelFunction {
    public:
        virtual double operator()(const VectorXd& x1, const VectorXd& x2) const = 0;
        virtual MatrixXd computeKernelMatrix(const MatrixXd& X1, const MatrixXd& X2) const = 0;
        virtual VectorXd computeKernelVector(const MatrixXd& X, const VectorXd& x) const = 0;
        virtual int getThetaDimension() const = 0;
        virtual ~KernelFunction() {} // Add a virtual destructor for proper polymorphic behavior

    };
    GaussianProcess(const MatrixXd& X, const VectorXd& y, const KernelFunction& kernel)
        : X_(X), y_(y), kernel_(kernel) {
        // Initialize hyperparameters, e.g., length scales, noise level
        // You may want to expose these as parameters or use hyperparameter optimization.
        theta_ = VectorXd::Ones(kernel.getThetaDimension());
        sigma_n_sq_ = 1e-6; // Small noise for numerical stability
        updateCovarianceMatrix();
    }

    // Add a new observation to the GP
    void addObservation(const VectorXd& x_new, double y_new) {
        X_.conservativeResize(X_.rows() + 1, X_.cols());
        X_.row(X_.rows() - 1) = x_new.transpose();
        y_.conservativeResize(y_.size() + 1);
        y_(y_.size() - 1) = y_new;
        updateCovarianceMatrix();
    }

    // Predict mean and variance at a new input point x
    std::pair<double, double> predict(const VectorXd& x) const {
        MatrixXd k_star = kernel_.computeKernelMatrix(X_, x);
        MatrixXd k_star_star = kernel_.computeKernelMatrix(x, x);
        VectorXd k_star_t = kernel_.computeKernelVector(X_, x);

        double mean = k_star_t.transpose() * K_inv_ * y_;
        double variance = k_star_star(0, 0) - (k_star_t.transpose() * K_inv_ * k_star).value();

        return std::make_pair(mean, std::max(variance, 0.0)); // Ensure variance is non-negative
    }

    // Acquisition functions
    double expectedImprovement(const VectorXd& x) const {
        auto [mean, variance] = predict(x);

        // Best observed value so far (maximum of training outputs)
        double best_observed = y_.maxCoeff();

        // Calculate the expected improvement
        double z = (mean - best_observed) / sqrt(variance);
        return (mean - best_observed) * NormalCDF(z) + sqrt(variance) * NormalPDF(z);
    }

    double upperConfidenceBound(const VectorXd& x, double kappa) const {
        auto [mean, variance] = predict(x);

        // Upper Confidence Bound (UCB) calculation
        return mean + kappa * sqrt(variance);
    }
private:
    // Update the covariance matrix K and its inverse K_inv when new observations are added
    void updateCovarianceMatrix() {
        K_ = kernel_.computeKernelMatrix(X_, X_) + sigma_n_sq_ * MatrixXd::Identity(X_.rows(), X_.rows());
        K_inv_ = K_.inverse();
    }
    VectorXd theta_;     // Hyperparameters of the kernel function
    // You can implement specific kernel functions by inheriting from KernelFunction
    // For example, here is a simple radial basis function (RBF) kernel
    class RBFKernel : public GaussianProcess::KernelFunction {
    public:
        double operator()(const VectorXd& x1, const VectorXd& x2) const override {
            double length_scale = exp(gp_.theta_(0)); // Length scale parameter
            return exp(-0.5 * (x1 - x2).squaredNorm() / (length_scale * length_scale));
        }

        MatrixXd computeKernelMatrix(const MatrixXd& X1, const MatrixXd& X2) const override {
            int n1 = X1.rows();
            int n2 = X2.rows();
            MatrixXd K(n1, n2);

            for (int i = 0; i < n1; ++i) {
                for (int j = 0; j < n2; ++j) {
                    K(i, j) = operator()(X1.row(i).transpose(), X2.row(j).transpose());
                }
            }

            return K;
        }

        VectorXd computeKernelVector(const MatrixXd& X, const VectorXd& x) const override {
            int n = X.rows();
            VectorXd k(n);

            for (int i = 0; i < n; ++i) {
                k(i) = operator()(X.row(i).transpose(), x);
            }

            return k;
        }

        int getThetaDimension() const override {
            return 1; // RBF kernel has one hyperparameter (length scale)
        }
        RBFKernel(const GaussianProcess& gp) : gp_(gp) {}

        private: 
        const GaussianProcess& gp_; // Reference to the associated GaussianProcess instance
    };

private:
    MatrixXd X_;         // Input data
    VectorXd y_;         // Output data
    MatrixXd K_;         // Covariance matrix
    MatrixXd K_inv_;     // Inverse of the covariance matrix
    KernelFunction kernel_; // Kernel function
    double sigma_n_sq_;  // Noise level
    double NormalPDF(double z) const {
        return exp(-0.5 * z * z) / sqrt(2.0 * M_PI);
    }

    double NormalCDF(double z) const {
        return 0.5 * (1.0 + erf(z / sqrt(2.0)));
    }
};
