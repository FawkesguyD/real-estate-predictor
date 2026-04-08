export type OpportunitySortBy = "score" | "undervaluation_percent";

export type Opportunity = {
  listing_id: number;
  title: string;
  city: string | null;
  district: string | null;
  area: number | null;
  rooms: number | null;
  floor: number | null;
  total_floors: number | null;
  listing_price: number | null;
  predicted_price: number;
  undervaluation_delta: number;
  undervaluation_percent: number;
  score: number;
  rank_position: number | null;
  is_saved: boolean;
};

export type OpportunityListResponse = {
  items: Opportunity[];
};
