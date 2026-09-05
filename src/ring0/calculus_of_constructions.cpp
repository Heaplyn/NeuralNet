#include "ring0/calculus_of_constructions.hpp"
#include <stdexcept>
#include <algorithm>

namespace ring0
{

    CoCTermPtr CoCTerm::make_universe(UniverseSort s, int lvl)
    {
        auto term = std::make_shared<CoCTerm>(TermKind::UNIVERSE);
        term->sort = s;
        term->level = lvl;
        return term;
    }

    CoCTermPtr CoCTerm::make_var(const std::string &name, int db_idx)
    {
        auto term = std::make_shared<CoCTerm>(TermKind::VARIABLE);
        term->var_name = name;
        term->de_bruijn_index = db_idx;
        return term;
    }

    CoCTermPtr CoCTerm::make_abstraction(const std::string &param, CoCTermPtr p_type, CoCTermPtr b)
    {
        auto term = std::make_shared<CoCTerm>(TermKind::ABSTRACTION);
        term->param_name = param;
        term->param_type = p_type;
        term->body = b;
        return term;
    }

    CoCTermPtr CoCTerm::make_application(CoCTermPtr f, CoCTermPtr a)
    {
        auto term = std::make_shared<CoCTerm>(TermKind::APPLICATION);
        term->func = f;
        term->arg = a;
        return term;
    }

    CoCTermPtr CoCTerm::make_pi(const std::string &param, CoCTermPtr p_type, CoCTermPtr b)
    {
        auto term = std::make_shared<CoCTerm>(TermKind::PI_TYPE);
        term->param_name = param;
        term->param_type = p_type;
        term->body = b;
        return term;
    }

    CoCTermPtr CoCTerm::make_arrow(CoCTermPtr from_type, CoCTermPtr to_type)
    {
        // Non-dependent function type: A -> B is shorthand for \Pi (_ : A). B
        return make_pi("_", from_type, to_type);
    }

    std::string CoCTerm::to_string() const
    {
        std::stringstream ss;
        switch (kind)
        {
        case TermKind::UNIVERSE:
            if (sort == UniverseSort::PROP)
                ss << "Prop";
            else
                ss << "Type_" << level;
            break;
        case TermKind::VARIABLE:
            ss << var_name;
            break;
        case TermKind::ABSTRACTION:
            ss << "(λ (" << param_name << " : "
               << (param_type ? param_type->to_string() : "Type") << "). "
               << (body ? body->to_string() : "nil") << ")";
            break;
        case TermKind::APPLICATION:
            ss << "(" << (func ? func->to_string() : "nil") << " "
               << (arg ? arg->to_string() : "nil") << ")";
            break;
        case TermKind::PI_TYPE:
            if (param_name == "_" || param_name.empty())
            {
                ss << "(" << (param_type ? param_type->to_string() : "nil") << " → "
                   << (body ? body->to_string() : "nil") << ")";
            }
            else
            {
                ss << "(Π (" << param_name << " : "
                   << (param_type ? param_type->to_string() : "Type") << "). "
                   << (body ? body->to_string() : "nil") << ")";
            }
            break;
        }
        return ss.str();
    }

    CoCTermPtr CoCTerm::substitute(const std::string &target_var, CoCTermPtr replacement) const
    {
        if (!replacement)
            return std::const_pointer_cast<CoCTerm>(shared_from_this());

        switch (kind)
        {
        case TermKind::UNIVERSE:
            return make_universe(sort, level);

        case TermKind::VARIABLE:
            if (var_name == target_var)
            {
                return replacement;
            }
            return make_var(var_name, de_bruijn_index);

        case TermKind::ABSTRACTION:
            if (param_name == target_var)
            {
                // Shadowed parameter: do not substitute inside body
                return make_abstraction(param_name,
                                        param_type ? param_type->substitute(target_var, replacement) : nullptr,
                                        body);
            }
            return make_abstraction(param_name,
                                    param_type ? param_type->substitute(target_var, replacement) : nullptr,
                                    body ? body->substitute(target_var, replacement) : nullptr);

        case TermKind::APPLICATION:
            return make_application(
                func ? func->substitute(target_var, replacement) : nullptr,
                arg ? arg->substitute(target_var, replacement) : nullptr);

        case TermKind::PI_TYPE:
            if (param_name == target_var)
            {
                return make_pi(param_name,
                               param_type ? param_type->substitute(target_var, replacement) : nullptr,
                               body);
            }
            return make_pi(param_name,
                           param_type ? param_type->substitute(target_var, replacement) : nullptr,
                           body ? body->substitute(target_var, replacement) : nullptr);
        }
        return nullptr;
    }

    CoCTermPtr CoCTerm::normalize_beta() const
    {
        switch (kind)
        {
        case TermKind::UNIVERSE:
        case TermKind::VARIABLE:
            return std::const_pointer_cast<CoCTerm>(shared_from_this());

        case TermKind::ABSTRACTION:
        {
            auto norm_type = param_type ? param_type->normalize_beta() : nullptr;
            auto norm_body = body ? body->normalize_beta() : nullptr;
            return make_abstraction(param_name, norm_type, norm_body);
        }

        case TermKind::APPLICATION:
        {
            auto norm_func = func ? func->normalize_beta() : nullptr;
            auto norm_arg = arg ? arg->normalize_beta() : nullptr;

            // Beta-Reduction Step: (\lambda x:A. b) a  ==>  b[x := a]
            if (norm_func && norm_func->kind == TermKind::ABSTRACTION && norm_func->body)
            {
                auto reduced = norm_func->body->substitute(norm_func->param_name, norm_arg);
                return reduced ? reduced->normalize_beta() : nullptr;
            }
            return make_application(norm_func, norm_arg);
        }

        case TermKind::PI_TYPE:
        {
            auto norm_type = param_type ? param_type->normalize_beta() : nullptr;
            auto norm_body = body ? body->normalize_beta() : nullptr;
            return make_pi(param_name, norm_type, norm_body);
        }
        }
        return std::const_pointer_cast<CoCTerm>(shared_from_this());
    }

    bool CoCTerm::is_definitionally_equal(const CoCTermPtr &other) const
    {
        if (!other)
            return false;
        if (this == other.get())
            return true;

        auto t1 = this->normalize_beta();
        auto t2 = other->normalize_beta();

        if (!t1 || !t2)
            return false;
        if (t1->kind != t2->kind)
            return false;

        switch (t1->kind)
        {
        case TermKind::UNIVERSE:
            return (t1->sort == t2->sort) && (t1->level == t2->level);

        case TermKind::VARIABLE:
            return t1->var_name == t2->var_name;

        case TermKind::ABSTRACTION:
        case TermKind::PI_TYPE:
        {
            bool param_match = (!t1->param_type && !t2->param_type) ||
                               (t1->param_type && t2->param_type && t1->param_type->is_definitionally_equal(t2->param_type));
            if (!param_match)
                return false;

            // Alpha-conversion check on bodies
            if (t1->param_name == t2->param_name)
            {
                return (!t1->body && !t2->body) || (t1->body && t2->body && t1->body->is_definitionally_equal(t2->body));
            }
            else if (t1->body && t2->body)
            {
                auto renamed_body2 = t2->body->substitute(t2->param_name, make_var(t1->param_name));
                return t1->body->is_definitionally_equal(renamed_body2);
            }
            return false;
        }

        case TermKind::APPLICATION:
            return (t1->func && t2->func && t1->func->is_definitionally_equal(t2->func)) &&
                   (t1->arg && t2->arg && t1->arg->is_definitionally_equal(t2->arg));
        }
        return false;
    }

    // =========================================================================
    // CoC Type-Checking Kernel Implementation
    // =========================================================================

    ProofValidationResult CoCTypeChecker::check_type(const TypingContext &ctx, const CoCTermPtr &term)
    {
        ProofValidationResult res;
        if (!term)
        {
            res.is_valid = false;
            res.error_message = "Null term encountered";
            return res;
        }

        switch (term->kind)
        {
        // ---------------------------------------------------------------------
        // Axiom: Universe Typing (Prop : Type_0, Type_i : Type_{i+1})
        // ---------------------------------------------------------------------
        case TermKind::UNIVERSE:
        {
            if (term->sort == UniverseSort::PROP)
            {
                res.is_valid = true;
                res.inferred_type = CoCTerm::make_universe(UniverseSort::TYPE_0, 0);
                res.proof_consistency_score = 1.0f;
                return res;
            }
            else
            {
                res.is_valid = true;
                res.inferred_type = CoCTerm::make_universe(UniverseSort::TYPE_1, term->level + 1);
                res.proof_consistency_score = 1.0f;
                return res;
            }
        }

        // ---------------------------------------------------------------------
        // Variable Rule: x : T where (x : T) in Gamma
        // ---------------------------------------------------------------------
        case TermKind::VARIABLE:
        {
            auto t = ctx.lookup(term->var_name);
            if (t)
            {
                res.is_valid = true;
                res.inferred_type = t;
                res.proof_consistency_score = 1.0f;
                return res;
            }
            res.is_valid = false;
            res.error_message = "Unbound variable in typing context: " + term->var_name;
            return res;
        }

        // ---------------------------------------------------------------------
        // Dependent Product Rule: \Pi (x : A). B(x)
        // In CoC: If A : s1 and B : s2, then \Pi(x:A).B : s2 (impredicative if s2 == Prop)
        // ---------------------------------------------------------------------
        case TermKind::PI_TYPE:
        {
            auto type_a_res = check_type(ctx, term->param_type);
            if (!type_a_res.is_valid)
            {
                res.is_valid = false;
                res.error_message = "Invalid domain type in Pi constructor: " + type_a_res.error_message;
                return res;
            }

            TypingContext extended_ctx = ctx;
            extended_ctx.extend(term->param_name, term->param_type);
            auto type_b_res = check_type(extended_ctx, term->body);
            if (!type_b_res.is_valid)
            {
                res.is_valid = false;
                res.error_message = "Invalid range type in Pi constructor: " + type_b_res.error_message;
                return res;
            }

            res.is_valid = true;
            // Universe sorting rule for Pi types:
            if (type_b_res.inferred_type && type_b_res.inferred_type->kind == TermKind::UNIVERSE &&
                type_b_res.inferred_type->sort == UniverseSort::PROP)
            {
                res.inferred_type = CoCTerm::make_universe(UniverseSort::PROP, 0); // Impredicativity of Prop
            }
            else
            {
                res.inferred_type = type_b_res.inferred_type ? type_b_res.inferred_type : CoCTerm::make_universe(UniverseSort::TYPE_0, 0);
            }
            res.proof_consistency_score = 1.0f;
            return res;
        }

        // ---------------------------------------------------------------------
        // Abstraction Rule: \lambda (x : A). b : \Pi (x : A). B(x)
        // ---------------------------------------------------------------------
        case TermKind::ABSTRACTION:
        {
            auto type_a_res = check_type(ctx, term->param_type);
            if (!type_a_res.is_valid)
            {
                res.is_valid = false;
                res.error_message = "Invalid parameter type in abstraction: " + type_a_res.error_message;
                return res;
            }

            TypingContext extended_ctx = ctx;
            extended_ctx.extend(term->param_name, term->param_type);
            auto body_res = check_type(extended_ctx, term->body);
            if (!body_res.is_valid)
            {
                res.is_valid = false;
                res.error_message = "Invalid body in abstraction: " + body_res.error_message;
                return res;
            }

            res.is_valid = true;
            res.inferred_type = CoCTerm::make_pi(term->param_name, term->param_type, body_res.inferred_type);
            res.proof_consistency_score = 1.0f;
            return res;
        }

        // ---------------------------------------------------------------------
        // Application Rule (Modus Ponens Elimination): (f a)
        // If f : \Pi (x : A). B(x) and a : A, then (f a) : B[x := a]
        // ---------------------------------------------------------------------
        case TermKind::APPLICATION:
        {
            auto func_res = check_type(ctx, term->func);
            if (!func_res.is_valid)
            {
                res.is_valid = false;
                res.error_message = "Function term in application is ill-typed: " + func_res.error_message;
                return res;
            }

            auto arg_res = check_type(ctx, term->arg);
            if (!arg_res.is_valid)
            {
                res.is_valid = false;
                res.error_message = "Argument term in application is ill-typed: " + arg_res.error_message;
                return res;
            }

            auto norm_func_type = func_res.inferred_type ? func_res.inferred_type->normalize_beta() : nullptr;
            if (!norm_func_type || norm_func_type->kind != TermKind::PI_TYPE)
            {
                res.is_valid = false;
                res.error_message = "Application target is not a function/Pi-type: " + (norm_func_type ? norm_func_type->to_string() : "nil");
                return res;
            }

            // Dependent type matching: domain of f must match arg type definitionally
            if (!norm_func_type->param_type->is_definitionally_equal(arg_res.inferred_type))
            {
                res.is_valid = false;
                res.error_message = "Type mismatch in application! Expected domain: " +
                                    norm_func_type->param_type->to_string() + " but got: " +
                                    arg_res.inferred_type->to_string();
                return res;
            }

            // Dependent Substitution: B[x := a]
            auto result_type = norm_func_type->body->substitute(norm_func_type->param_name, term->arg);
            res.is_valid = true;
            res.inferred_type = result_type ? result_type->normalize_beta() : nullptr;
            res.proof_consistency_score = 1.0f;
            return res;
        }
        }
        res.is_valid = false;
        res.error_message = "Unknown term kind";
        return res;
    }

    ProofValidationResult CoCTypeChecker::verify_proof(
        const TypingContext &ctx,
        const CoCTermPtr &witness,
        const CoCTermPtr &expected_prop)
    {
        auto type_res = check_type(ctx, witness);
        if (!type_res.is_valid)
        {
            return type_res;
        }

        if (!expected_prop)
        {
            type_res.proof_consistency_score = 0.85f;
            return type_res;
        }

        // Check if inferred type definitionally proves expected proposition: T \equiv expected_prop
        if (type_res.inferred_type->is_definitionally_equal(expected_prop))
        {
            type_res.is_valid = true;
            type_res.proof_consistency_score = 1.0f;
            return type_res;
        }

        ProofValidationResult mismatch_res;
        mismatch_res.is_valid = false;
        mismatch_res.error_message = "Proof witness proved: " + type_res.inferred_type->to_string() +
                                     " but goal was: " + expected_prop->to_string();
        mismatch_res.inferred_type = type_res.inferred_type;
        mismatch_res.proof_consistency_score = 0.2f;
        return mismatch_res;
    }

    TypingContext CoCTypeChecker::create_standard_logic_context()
    {
        TypingContext ctx;
        auto prop = CoCTerm::make_universe(UniverseSort::PROP, 0);
        auto type0 = CoCTerm::make_universe(UniverseSort::TYPE_0, 0);

        // Standard logic symbols
        ctx.extend("Prop", type0);
        ctx.extend("Type", CoCTerm::make_universe(UniverseSort::TYPE_1, 1));
        ctx.extend("Truth", prop);
        ctx.extend("Falsity", prop);

        // Classical Identity Type: Id : \Pi (A : Type). A -> A -> Prop
        auto id_type = CoCTerm::make_pi("A", type0,
                            CoCTerm::make_arrow(CoCTerm::make_var("A"),
                                CoCTerm::make_arrow(CoCTerm::make_var("A"), prop)));
        ctx.extend("Id", id_type);

        return ctx;
    }

} // namespace ring0
