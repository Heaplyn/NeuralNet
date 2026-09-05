#pragma once

#include <string>
#include <vector>
#include <memory>
#include <map>
#include <iostream>
#include <sstream>

namespace ring0
{

    /**
     * @enum UniverseLevel
     * @brief Sorts and Universe stratification in Calculus of Constructions (CoC).
     * Prop is the impredicative universe of propositions/proofs.
     * Type_k are predicative universes of computational data and types.
     */
    enum class UniverseSort
    {
        PROP = 0,  ///< Propositional universe (impredicative)
        TYPE_0 = 1,///< Concrete data types (tensors, vectors, tokens)
        TYPE_1 = 2,///< Module and layer type constructors
        TYPE_2 = 3 ///< Architecture and meta-policy universe
    };

    /**
     * @enum TermKind
     * @brief AST Node kinds in the Lambda Cube / CoC.
     */
    enum class TermKind
    {
        UNIVERSE,     ///< Prop or Type_i
        VARIABLE,     ///< x (named variable)
        ABSTRACTION,  ///< \lambda (x : A). b (Function abstraction / proof constructor)
        APPLICATION,  ///< (f a) (Function application / proof elimination via Modus Ponens)
        PI_TYPE       ///< \Pi (x : A). B(x) (Dependent Product: A -> B, forall x:A. B(x), or polymorphic Pi X:Type. T)
    };

    class CoCTerm;
    using CoCTermPtr = std::shared_ptr<CoCTerm>;

    /**
     * @class CoCTerm
     * @brief Abstract Syntax Tree node for terms and types in the Calculus of Constructions.
     */
    class CoCTerm : public std::enable_shared_from_this<CoCTerm>
    {
    public:
        TermKind kind;

        // Universe payload
        UniverseSort sort = UniverseSort::PROP;
        int level = 0;

        // Variable payload
        std::string var_name = "";
        int de_bruijn_index = -1;

        // Abstraction & Pi-Type payload
        std::string param_name = "";
        CoCTermPtr param_type = nullptr;
        CoCTermPtr body = nullptr;

        // Application payload
        CoCTermPtr func = nullptr;
        CoCTermPtr arg = nullptr;

        explicit CoCTerm(TermKind k) : kind(k) {}

        // --- Static Factory Constructors ---
        static CoCTermPtr make_universe(UniverseSort s, int lvl = 0);
        static CoCTermPtr make_var(const std::string &name, int db_idx = -1);
        static CoCTermPtr make_abstraction(const std::string &param, CoCTermPtr p_type, CoCTermPtr b);
        static CoCTermPtr make_application(CoCTermPtr f, CoCTermPtr a);
        static CoCTermPtr make_pi(const std::string &param, CoCTermPtr p_type, CoCTermPtr b);
        static CoCTermPtr make_arrow(CoCTermPtr from_type, CoCTermPtr to_type);

        /// Returns human-readable string representation of the CoC expression
        std::string to_string() const;

        /// Performs capture-avoiding substitution: this[var_name := replacement]
        CoCTermPtr substitute(const std::string &target_var, CoCTermPtr replacement) const;

        /// Evaluates term to weak head normal form (Beta-reduction)
        CoCTermPtr normalize_beta() const;

        /// Evaluates alpha-beta definitional equivalence between two terms
        bool is_definitionally_equal(const CoCTermPtr &other) const;
    };

    /**
     * @struct TypingContext
     * @brief Context environment Gamma containing variable-to-type bindings (x : T).
     */
    struct TypingContext
    {
        std::map<std::string, CoCTermPtr> bindings;

        void extend(const std::string &var, CoCTermPtr type)
        {
            bindings[var] = type;
        }

        CoCTermPtr lookup(const std::string &var) const
        {
            auto it = bindings.find(var);
            if (it != bindings.end())
                return it->second;
            return nullptr;
        }
    };

    /**
     * @struct ProofValidationResult
     * @brief Formal verification telemetry produced by the CoC Type Checker Kernel.
     */
    struct ProofValidationResult
    {
        bool is_valid = false;               ///< True if term is well-typed (witness proves proposition)
        std::string error_message = "";      ///< Diagnostic explanation if type check fails
        CoCTermPtr inferred_type = nullptr;  ///< Inferred type / proved proposition T
        size_t reduction_steps = 0;          ///< Total beta-reduction steps during normalization
        float proof_consistency_score = 0.0f;///< Normalized proof validity confidence score [0.0, 1.0]
    };

    /**
     * @class CoCTypeChecker
     * @brief Decidable Type Checker and Proof Verification Kernel for the Calculus of Constructions.
     */
    class CoCTypeChecker
    {
    public:
        /// Infers and verifies the type of term t in context Gamma: Gamma |- t : T
        static ProofValidationResult check_type(const TypingContext &ctx, const CoCTermPtr &term);

        /// Verifies whether term witness proves goal proposition: Gamma |- witness : expected_proposition
        static ProofValidationResult verify_proof(const TypingContext &ctx, const CoCTermPtr &witness, const CoCTermPtr &expected_prop);

        /// Constructs standard logical axioms (Modus Ponens, Identity, Transitivity, Type-of-Types)
        static TypingContext create_standard_logic_context();
    };

} // namespace ring0
